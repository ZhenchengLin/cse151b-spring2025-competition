# -*- coding: utf-8 -*-
"""
Enhanced Climate Emulation Model for CSE151B Spring 2025 Competition
Implements advanced strategies from the strategic plan including:
- Sophisticated data preprocessing with climatology removal
- Advanced architectures with FiLM layers for scalar-spatial fusion
- Multi-component loss functions with physical constraints
- Ensemble strategies for robust predictions
"""

import os
import numpy as np
import xarray as xr
import dask.array as da
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import PowerTransformer
import warnings
warnings.filterwarnings('ignore')
import cftime

config = {
    "data": {
        "path": "processed_data_cse151b_v2_corrupted_ssp245/processed_data_cse151b_v2_corrupted_ssp245.zarr",
        "input_vars": ["CO2", "SO2", "CH4", "BC", "rsdt"],
        "output_vars": ["tas", "pr"],
        "target_member_id": 0,
        "train_ssps": ["ssp126", "ssp370", "ssp585"],
        "test_ssp": "ssp245",
        "test_months": 360,
        "batch_size": 32,
        "num_workers": 4,
        "use_climatology_removal": True,
        "reference_period": slice("2015", "2045"),
    },
    "model": {
        "type": "enhanced_unet_film",
        "encoder_channels": [64, 128, 256, 512],
        "decoder_channels": [256, 128, 64],
        "film_hidden_dim": 256,
        "use_attention": True,
        "dropout_rate": 0.1,
    },
    "training": {
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "loss_weights": {
            "rmse": 1.0,
            "decadal_mean": 0.1,
            "decadal_std": 0.1,
            "pr_nonneg": 0.01,
        },
        "multi_step_training": True,
        "rollout_steps": 3,
    },
    "trainer": {
        "max_epochs": 50,
        "accelerator": "auto",
        "devices": "auto",
        "precision": 16,
        "deterministic": True,
        "num_sanity_val_steps": 0,
        "callbacks": [],
    },
    "seed": 42,
}

pl.seed_everything(config["seed"])

def get_lat_weights(latitude_values):
    lat_rad = np.deg2rad(latitude_values)
    weights = np.cos(lat_rad)
    return weights / (np.mean(weights) + 1e-9)

def encode_cyclical_features(time_data, period='year'):
    if period == 'year':
        values = time_data.dayofyear / 365.0
    elif period == 'month':
        actual_month_numbers = None
        if hasattr(time_data, 'month'):
            actual_month_numbers = time_data.month
        elif isinstance(time_data, pd.Series):
            actual_month_numbers = time_data.values
        elif isinstance(time_data, np.ndarray):
            actual_month_numbers = time_data
        else:
            raise TypeError(f"Unsupported type for time_data in 'month' period: {type(time_data)}")

        if actual_month_numbers is None:
            raise ValueError("Could not extract month numbers for cyclical encoding.")

        values = actual_month_numbers / 12.0
    else:
        raise ValueError(f"Unknown period: {period}")

    sin_values = np.sin(2 * np.pi * values)
    cos_values = np.cos(2 * np.pi * values)
    return sin_values, cos_values

class Normalizer:
    def __init__(self):
        self.mean_in, self.std_in = None, None
        self.mean_out, self.std_out = None, None

    def set_input_statistics(self, mean, std):
        self.mean_in = mean
        self.std_in = std
        if self.std_in is not None and np.any(np.isclose(self.std_in, 0)):
            print("WARNING: Input standard deviation is close to zero!")
        if self.std_in is not None and np.any(self.std_in <= 1e-9):
            print(f"WARNING: Input standard deviation has very small values <= 1e-9.")

    def set_output_statistics(self, mean, std):
        self.mean_out = mean
        self.std_out = std
        if self.std_out is not None and np.any(np.isclose(self.std_out, 0)):
            print("WARNING: Output standard deviation is close to zero!")
        if self.std_out is not None and np.any(self.std_out <= 1e-9):
            print(f"WARNING: Output standard deviation has very small values <= 1e-9.")

    def normalize(self, data, data_type):
        epsilon = 1e-9
        if data_type == "input":
            if self.mean_in is None or self.std_in is None:
                raise ValueError("Input statistics not set for Normalizer.")
            return (data - self.mean_in) / (self.std_in + epsilon)
        elif data_type == "output":
            if self.mean_out is None or self.std_out is None:
                raise ValueError("Output statistics not set for Normalizer.")
            return (data - self.mean_out) / (self.std_out + epsilon)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

    def inverse_transform_output(self, data):
        epsilon = 1e-9
        if self.mean_out is None or self.std_out is None:
            raise ValueError("Output statistics not set for Normalizer.")
        return data * (self.std_out + epsilon) + self.mean_out

class FiLMLayer(nn.Module):
    def __init__(self, num_features, condition_dim, hidden_dim=256):
        super().__init__()
        self.num_features = num_features
        self.film_generator = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * num_features)
        )

    def forward(self, features, condition):
        params = self.film_generator(condition)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1)
        return gamma * features + beta

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        num_groups = 8
        if channels > 0 and channels % num_groups != 0:
            for ng_candidate in range(num_groups, 0, -1):
                if channels % ng_candidate == 0:
                    num_groups = ng_candidate
                    break
            if channels % num_groups != 0:
                num_groups = 1

        self.norm = nn.GroupNorm(num_groups, channels) if channels > 0 else nn.Identity()
        self.q = nn.Conv2d(channels, channels, 1) if channels > 0 else nn.Identity()
        self.k = nn.Conv2d(channels, channels, 1) if channels > 0 else nn.Identity()
        self.v = nn.Conv2d(channels, channels, 1) if channels > 0 else nn.Identity()
        self.proj_out = nn.Conv2d(channels, channels, 1) if channels > 0 else nn.Identity()

    def forward(self, x):
        if self.channels == 0: return x
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w).permute(0, 2, 1)
        k = k.reshape(b, c, h*w)
        attn_scale = c ** -0.5 if c > 0 else 1.0
        attn = torch.bmm(q, k) * attn_scale
        attn = F.softmax(attn, dim=2)

        v = v.reshape(b, c, h*w).permute(0, 2, 1)
        h_ = torch.bmm(attn, v)
        h_ = h_.permute(0, 2, 1).reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        return x + h_

class EnhancedUNetWithFiLM(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, n_scalar_inputs,
                 encoder_channels=[64, 128, 256, 512],
                 decoder_channels=[256, 128, 64],
                 film_hidden_dim=256, use_attention=True, dropout_rate=0.1):
        super().__init__()

        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels

        if len(self.decoder_channels) != len(self.encoder_channels) -1 :
             print(f"CRITICAL WARNING: Length of decoder_channels ({len(self.decoder_channels)}) "
                  f"does not match expected ({len(self.encoder_channels)-1}). This will likely cause errors.")

        self.scalar_embed = nn.Sequential(
            nn.Linear(n_scalar_inputs, film_hidden_dim),
            nn.ReLU(),
            nn.Linear(film_hidden_dim, film_hidden_dim)
        )

        self.encoder_convs = nn.ModuleList()
        self.encoder_films = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()

        in_channels = n_input_channels
        for i, out_channels in enumerate(encoder_channels):
            self.encoder_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels), nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels), nn.ReLU()
            ))
            self.encoder_films.append(FiLMLayer(out_channels, film_hidden_dim, film_hidden_dim))
            if i < len(encoder_channels) - 1:
                self.encoder_pools.append(nn.MaxPool2d(2))
            in_channels = out_channels

        if use_attention and encoder_channels[-1] > 0 :
            self.attention = AttentionBlock(encoder_channels[-1])
        else:
            self.attention = nn.Identity()

        self.decoder_upsamples = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        self.decoder_films = nn.ModuleList()

        n_encoder_layers = len(self.encoder_channels)
        for i in range(n_encoder_layers - 1):
            in_channels_up = self.encoder_channels[-1] if i == 0 else self.decoder_channels[i-1]
            skip_connection_channels = self.encoder_channels[n_encoder_layers - 2 - i]
            up_channels_out = skip_connection_channels

            self.decoder_upsamples.append(nn.ConvTranspose2d(
                in_channels_up, up_channels_out, kernel_size=2, stride=2
            ))
            in_channels_conv = up_channels_out + skip_connection_channels
            out_channels_conv = self.decoder_channels[i]
            self.decoder_convs.append(nn.Sequential(
                nn.Conv2d(in_channels_conv, out_channels_conv, 3, padding=1),
                nn.BatchNorm2d(out_channels_conv), nn.ReLU(),
                nn.Conv2d(out_channels_conv, out_channels_conv, 3, padding=1),
                nn.BatchNorm2d(out_channels_conv), nn.ReLU()
            ))
            self.decoder_films.append(FiLMLayer(out_channels_conv, film_hidden_dim, film_hidden_dim))

        self.dropout = nn.Dropout2d(dropout_rate)
        self.output_conv = nn.Conv2d(self.decoder_channels[-1], n_output_channels, 1)

    def forward(self, x_spatial, x_scalar):
        scalar_embed = self.scalar_embed(x_scalar)
        encoder_features = []
        x = x_spatial
        for i in range(len(self.encoder_convs)):
            x = self.encoder_convs[i](x)
            x = self.encoder_films[i](x, scalar_embed)
            encoder_features.append(x)
            if i < len(self.encoder_pools):
                x = self.encoder_pools[i](x)

        x = self.attention(x)

        for i in range(len(self.decoder_upsamples)):
            x = self.decoder_upsamples[i](x)
            skip_connection_source_index = len(encoder_features) - 2 - i
            if 0 <= skip_connection_source_index < len(encoder_features):
                skip_feature = encoder_features[skip_connection_source_index]
                if x.shape[2:] != skip_feature.shape[2:]:
                    x = F.interpolate(x, size=skip_feature.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip_feature], dim=1)
            else:
                print(f"ERROR DECODER i={i}: Invalid skip_connection_source_index: {skip_connection_source_index}. len(encoder_features)={len(encoder_features)}")
            x = self.decoder_convs[i](x)
            x = self.decoder_films[i](x, scalar_embed)

        x = self.dropout(x)
        output = self.output_conv(x)
        if output.shape[2:] != x_spatial.shape[2:]:
            output = F.interpolate(output, size=x_spatial.shape[2:], mode='bilinear', align_corners=False)
        tas, pr = output.split(1, dim=1)
        pr = F.relu(pr)
        if torch.isnan(tas).any() or torch.isnan(pr).any():
            print("NaNs detected in final tas/pr output of the model forward pass!")
        return torch.cat([tas, pr], dim=1)

class EnhancedClimateDataset(Dataset):
    def __init__(self, inputs_spatial, inputs_scalar, outputs,
                 output_is_normalized=True, transform_pr=None):
        self.size = inputs_spatial.shape[0]
        self.inputs_spatial = torch.from_numpy(inputs_spatial).float()
        self.inputs_scalar = torch.from_numpy(inputs_scalar).float()
        self.outputs = torch.from_numpy(outputs).float()

        for name, tensor_data in [("inputs_spatial", self.inputs_spatial),
                                  ("inputs_scalar", self.inputs_scalar),
                                  ("outputs", self.outputs)]:
            if torch.isnan(tensor_data).any():
                nan_indices = torch.nonzero(torch.isnan(tensor_data))
                print(f"NaNs found in dataset: {name} at first 5 indices: {nan_indices[:5] if nan_indices.numel() > 0 else 'None'}")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.inputs_spatial[idx], self.inputs_scalar[idx], self.outputs[idx]

class EnhancedClimateDataModule(pl.LightningDataModule):
    def __init__(self, path, input_vars, output_vars, train_ssps, test_ssp,
                 target_member_id, test_months=120, batch_size=32, num_workers=0,
                 use_climatology_removal=True, reference_period=None, seed=42):
        super().__init__()
        self.path = path
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.train_ssps = train_ssps
        self.test_ssp = test_ssp
        self.target_member_id = target_member_id
        self.test_months = test_months
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_climatology_removal = use_climatology_removal
        self.reference_period = reference_period
        self.seed = seed
        self.normalizer = Normalizer()
        self.pr_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
        self.n_train_samples_for_clim_ref = 0

    def setup(self, stage=None):
        print(f"DEBUG: Attempting to open Zarr store: {self.path}")
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Zarr store not found at {self.path}.")
        ds = xr.open_zarr(self.path, consolidated=False, chunks={"time": "auto"})

        print("--- DEBUG: Checking raw data for NaNs ---")
        for var_name in self.input_vars + self.output_vars:
            if var_name in ds:
                for ssp_check in self.train_ssps + [self.test_ssp]:
                    try:
                        data_to_check = ds[var_name].sel(ssp=ssp_check)
                        if "member_id" in data_to_check.dims:
                            data_to_check = data_to_check.sel(member_id=self.target_member_id, drop=True)
                        if da.isnull(data_to_check.data).any().compute():
                            print(f"WARNING: NaNs found in raw data for variable '{var_name}' in SSP '{ssp_check}'!")
                    except Exception as e:
                        print(f"Could not check var {var_name} for ssp {ssp_check}. Error: {e}")
            else:
                print(f"Warning: Variable '{var_name}' not found in Zarr dataset.")
        print("--- DEBUG: Raw data check complete ---")

        def load_ssp(ssp_name_local):
            spatial_inputs_list, scalar_inputs_list = [], []
            for var in self.input_vars:
                if var not in ds: continue
                da_var = ds[var].sel(ssp=ssp_name_local)
                if "member_id" in da_var.dims: da_var = da_var.sel(member_id=self.target_member_id, drop=True)
                if len(da_var.dims) > 1 and "time" in da_var.dims: spatial_inputs_list.append(da_var.data)
                elif len(da_var.dims) == 1 and "time" in da_var.dims: scalar_inputs_list.append(da_var.data[:, np.newaxis])

            ref_var_name_for_time = self.output_vars[0] if self.output_vars and self.output_vars[0] in ds else \
                                    self.input_vars[0] if self.input_vars and self.input_vars[0] in ds else None
            if not ref_var_name_for_time: raise ValueError(f"Cannot determine time dimension for SSP {ssp_name_local}")
            time_ref_data = ds[ref_var_name_for_time].sel(ssp=ssp_name_local)
            if "member_id" in time_ref_data.dims: time_ref_data = time_ref_data.sel(member_id=self.target_member_id, drop=True)
            n_times_ssp = time_ref_data.sizes["time"]
            n_y_ssp = time_ref_data.sizes.get("y", ds[self.output_vars[0]].sel(ssp=ssp_name_local, member_id=self.target_member_id).sizes.get("y", 48))
            n_x_ssp = time_ref_data.sizes.get("x", ds[self.output_vars[0]].sel(ssp=ssp_name_local, member_id=self.target_member_id).sizes.get("x", 72))

            spatial_input_stack = da.stack(spatial_inputs_list, axis=1) if spatial_inputs_list else \
                                  da.zeros((n_times_ssp, 0, n_y_ssp, n_x_ssp), chunks={"time":"auto", 0:1, "y":-1, "x":-1 })

            time_values_raw = time_ref_data.time.values
            if isinstance(time_values_raw[0], cftime.datetime):
                months_enc = np.array([t.month for t in time_values_raw])
                years_enc = np.array([t.year for t in time_values_raw])
            else:
                try:
                    time_index_pd = pd.to_datetime(time_values_raw)
                    months_enc = time_index_pd.month.values
                    years_enc = time_index_pd.year.values
                except Exception as e:
                    print(f"Fallback time conversion for SSP {ssp_name_local}, error: {e}.")
                    months_enc = np.array([getattr(t, 'month', 1) for t in time_values_raw])
                    years_enc = np.array([getattr(t, 'year', 2000) for t in time_values_raw])

            month_sin, month_cos = encode_cyclical_features(months_enc, period='month')
            min_year, max_year = years_enc.min(), years_enc.max()
            year_norm = (years_enc - min_year) / (max_year - min_year + 1e-9) if max_year > min_year else np.zeros_like(years_enc)

            scalar_inputs_list.extend([da.from_array(f[:, np.newaxis], chunks=("auto", 1)) for f in [month_sin, month_cos, year_norm]])
            scalar_input_stack = da.concatenate(scalar_inputs_list, axis=1) if scalar_inputs_list else \
                                 da.zeros((n_times_ssp, 0), chunks=("auto",1))

            outputs_list = [ds[var].sel(ssp=ssp_name_local, member_id=self.target_member_id).data for var in self.output_vars if var in ds]
            if not outputs_list: raise ValueError(f"No output variables for SSP {ssp_name_local}")
            output_stack = da.stack(outputs_list, axis=1)
            return spatial_input_stack, scalar_input_stack, output_stack

        train_spatial_ssp, train_scalar_ssp, train_output_ssp = [], [], []
        for ssp_name_outer in self.train_ssps:
            spatial, scalar, output = load_ssp(ssp_name_outer)
            if ssp_name_outer == "ssp370":
                val_spatial, val_scalar, val_output = spatial[-self.test_months:], scalar[-self.test_months:], output[-self.test_months:]
                train_spatial_ssp.append(spatial[:-self.test_months]); train_scalar_ssp.append(scalar[:-self.test_months]); train_output_ssp.append(output[:-self.test_months])
            else:
                train_spatial_ssp.append(spatial); train_scalar_ssp.append(scalar); train_output_ssp.append(output)

        train_spatial_all = da.concatenate(train_spatial_ssp, axis=0)
        train_scalar_all = da.concatenate(train_scalar_ssp, axis=0)
        train_output_all = da.concatenate(train_output_ssp, axis=0)
        self.n_train_samples_for_clim_ref = train_output_all.shape[0]

        if self.use_climatology_removal:
            print("DEBUG: Applying climatology removal...")
            train_output_np = train_output_all.compute()
            n_s, n_v, n_y, n_x = train_output_np.shape
            self.output_climatology = np.zeros((12, n_v, n_y, n_x))
            for m_idx in range(12):
                mask = (np.arange(n_s) % 12) == m_idx
                if np.any(mask): self.output_climatology[m_idx] = np.nanmean(train_output_np[mask], axis=0)
            self.output_climatology = np.nan_to_num(self.output_climatology, nan=0.0)

            train_anom_np = np.array([train_output_np[i] - self.output_climatology[i % 12] for i in range(n_s)])
            train_output_processed = da.from_array(train_anom_np, chunks=train_output_all.chunks)

            if val_output is not None:
                val_output_np = val_output.compute()
                val_anom_np = np.array([val_output_np[i] - self.output_climatology[(self.n_train_samples_for_clim_ref + i)%12] for i in range(val_output_np.shape[0])])
                val_output_processed = da.from_array(val_anom_np, chunks=val_output.chunks)
            else: val_output_processed = None
        else:
            train_output_processed, val_output_processed, self.output_climatology = train_output_all, val_output, None

        epsilon_std = 1e-7
        if train_spatial_all.shape[1] > 0:
            mean_in_s, std_in_s = da.nanmean(train_spatial_all, axis=(0,2,3), keepdims=True).compute(), da.nanstd(train_spatial_all, axis=(0,2,3), keepdims=True).compute()
            std_in_s[std_in_s < epsilon_std] = epsilon_std
            self.normalizer.set_input_statistics(mean_in_s, std_in_s)
        else: self.normalizer.set_input_statistics(np.empty((1,0,1,1)), np.empty((1,0,1,1)))

        if train_scalar_all.shape[1] > 0:
            self.scalar_mean, self.scalar_std = da.nanmean(train_scalar_all, axis=0, keepdims=True).compute(), da.nanstd(train_scalar_all, axis=0, keepdims=True).compute()
            self.scalar_std[self.scalar_std < epsilon_std] = epsilon_std
        else: self.scalar_mean, self.scalar_std = np.empty((1,0)), np.empty((1,0))

        train_output_for_norm_np = train_output_processed.compute()
        if "pr" in self.output_vars:
            try:
                pr_idx = self.output_vars.index("pr")
                pr_data_train_yj = train_output_for_norm_np[:, pr_idx].reshape(-1,1)
                if np.isnan(pr_data_train_yj).any(): pr_data_train_yj = np.nan_to_num(pr_data_train_yj, nan=np.nanmedian(pr_data_train_yj))
                self.pr_transformer.fit(pr_data_train_yj)
                transformed_pr = self.pr_transformer.transform(pr_data_train_yj).reshape(train_output_for_norm_np.shape[0], train_output_for_norm_np.shape[2], train_output_for_norm_np.shape[3])
                train_output_for_norm_np[:, pr_idx] = transformed_pr
            except Exception as e: print(f"Error during Yeo-Johnson for training 'pr': {e}")

        mean_out_final, std_out_final = np.nanmean(train_output_for_norm_np, axis=(0,2,3), keepdims=True), np.nanstd(train_output_for_norm_np, axis=(0,2,3), keepdims=True)
        std_out_final[std_out_final < epsilon_std] = epsilon_std
        self.normalizer.set_output_statistics(mean_out_final, std_out_final)

        train_spatial_norm_da = self.normalizer.normalize(train_spatial_all, "input") if train_spatial_all.shape[1] > 0 else train_spatial_all
        train_scalar_norm_da = (train_scalar_all - self.scalar_mean) / (self.scalar_std + epsilon_std) if train_scalar_all.shape[1] > 0 else train_scalar_all
        train_output_norm_da = self.normalizer.normalize(da.from_array(train_output_for_norm_np, chunks=train_output_processed.chunks), "output")

        if val_output_processed is not None:
            val_spatial_norm_da = self.normalizer.normalize(val_spatial, "input") if val_spatial.shape[1] > 0 else val_spatial
            val_scalar_norm_da = (val_scalar - self.scalar_mean) / (self.scalar_std + epsilon_std) if val_scalar.shape[1] > 0 else val_scalar
            val_output_for_norm_np = val_output_processed.compute()
            if "pr" in self.output_vars:
                try:
                    pr_idx_val = self.output_vars.index("pr")
                    pr_data_val_yj = val_output_for_norm_np[:, pr_idx_val].reshape(-1,1)
                    if np.isnan(pr_data_val_yj).any(): pr_data_val_yj = np.nan_to_num(pr_data_val_yj, nan=0)
                    transformed_pr_val = self.pr_transformer.transform(pr_data_val_yj).reshape(val_output_for_norm_np.shape[0], val_output_for_norm_np.shape[2], val_output_for_norm_np.shape[3])
                    val_output_for_norm_np[:, pr_idx_val] = transformed_pr_val
                except Exception as e: print(f"Error during Yeo-Johnson for validation 'pr': {e}")
            val_output_norm_da = self.normalizer.normalize(da.from_array(val_output_for_norm_np, chunks=val_output_processed.chunks), "output")
        else: val_spatial_norm_da, val_scalar_norm_da, val_output_norm_da = None, None, None

        test_spatial_raw, test_scalar_raw, test_output_raw_for_eval = load_ssp(self.test_ssp)
        test_spatial_raw,test_scalar_raw,test_output_raw_for_eval = test_spatial_raw[-self.test_months:],test_scalar_raw[-self.test_months:],test_output_raw_for_eval[-self.test_months:]
        test_spatial_norm_da = self.normalizer.normalize(test_spatial_raw, "input") if test_spatial_raw.shape[1] > 0 else test_spatial_raw
        test_scalar_norm_da = (test_scalar_raw - self.scalar_mean) / (self.scalar_std + epsilon_std) if test_scalar_raw.shape[1] > 0 else test_scalar_raw

        def dask_to_numpy_safe(dask_array, name):
            if dask_array is None: return None
            computed_array = dask_array.compute()
            if np.isnan(computed_array).any():
                print(f"CRITICAL WARNING: NaNs in {name} after compute()! Sum: {np.sum(np.isnan(computed_array))}")
            return computed_array

        train_s_np, train_sc_np, train_o_np = dask_to_numpy_safe(train_spatial_norm_da, "train_spatial_norm"), dask_to_numpy_safe(train_scalar_norm_da, "train_scalar_norm"), dask_to_numpy_safe(train_output_norm_da, "train_output_norm")
        self.train_dataset = EnhancedClimateDataset(train_s_np, train_sc_np, train_o_np)

        if val_spatial_norm_da is not None:
            val_s_np, val_sc_np, val_o_np = dask_to_numpy_safe(val_spatial_norm_da, "val_spatial_norm"), dask_to_numpy_safe(val_scalar_norm_da, "val_scalar_norm"), dask_to_numpy_safe(val_output_norm_da, "val_output_norm")
            self.val_dataset = EnhancedClimateDataset(val_s_np, val_sc_np, val_o_np)
        else: self.val_dataset = None

        test_s_np, test_sc_np, test_o_raw_np = dask_to_numpy_safe(test_spatial_norm_da, "test_spatial_norm"), dask_to_numpy_safe(test_scalar_norm_da, "test_scalar_norm"), dask_to_numpy_safe(test_output_raw_for_eval, "test_output_raw_for_eval")
        self.test_dataset = EnhancedClimateDataset(test_s_np, test_sc_np, test_o_raw_np, output_is_normalized=False)

        try:
            template_slice_isel = {"time": 0}
            spatial_template_var_name = "rsdt"
            if spatial_template_var_name not in ds or len(ds[spatial_template_var_name].sel(ssp=self.train_ssps[0]).isel(**template_slice_isel).shape) < 2:
                spatial_template_var_name = self.output_vars[0] if self.output_vars and self.output_vars[0] in ds else self.input_vars[0]

            spatial_template_da_ssp = ds[spatial_template_var_name].sel(ssp=self.train_ssps[0])
            spatial_template_da = spatial_template_da_ssp.isel(**template_slice_isel)

            if "member_id" in spatial_template_da.dims:
                spatial_template_da = spatial_template_da.sel(member_id=self.target_member_id, drop=True)

            if len(spatial_template_da.shape) != 2:
                raise ValueError(f"Spatial template for {spatial_template_var_name} is not 2D after selections. Shape: {spatial_template_da.shape}, Dims: {spatial_template_da.dims}")

            self.lat = spatial_template_da.y.values if "y" in spatial_template_da.coords else spatial_template_da.latitude.values
            self.lon = spatial_template_da.x.values if "x" in spatial_template_da.coords else spatial_template_da.longitude.values
            self.area_weights = xr.DataArray(get_lat_weights(self.lat), dims=["y"], coords={"y": self.lat})
            print(f"DEBUG: Successfully extracted lat/lon from '{spatial_template_var_name}'.")
        except Exception as e:
            print(f"Error getting lat/lon: {e}. Using dummy.")
            self.lat, self.lon = np.linspace(-88.75,88.75,48), np.linspace(0,357.5,72)
            self.area_weights = xr.DataArray(get_lat_weights(self.lat), dims=["y"], coords={"y": self.lat})
        print("--- DEBUG: DataModule setup complete. ---")

    def train_dataloader(self):
        if not hasattr(self, 'train_dataset') or self.train_dataset is None: raise RuntimeError("train_dataset not init.")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        if not hasattr(self, 'val_dataset') or self.val_dataset is None:
             class EmptyDataset(Dataset):
                 def __len__(self): return 0
                 def __getitem__(self, idx): raise IndexError
             return DataLoader(EmptyDataset(), batch_size=self.batch_size)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, drop_last=True)

    def test_dataloader(self):
        if not hasattr(self, 'test_dataset') or self.test_dataset is None: raise RuntimeError("test_dataset not init.")
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, drop_last=False)

class EnhancedClimateEmulationModule(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-4, weight_decay=1e-5, loss_weights=None, **kwargs):
        super().__init__()
        self.model = model
        hparams_to_save = {
            "learning_rate": learning_rate, "weight_decay": weight_decay,
            "loss_weights": loss_weights or {"rmse":1.0,"decadal_mean":0.1,"decadal_std":0.1,"pr_nonneg":0.01}
        }
        self.save_hyperparameters(hparams_to_save)
        self.loss_weights = self.hparams.loss_weights
        self.normalizer, self.area_weights_xr, self.output_climatology, self.pr_transformer, self.dm_output_vars = None, None, None, None, []
        self.val_preds, self.val_targets, self.test_preds, self.test_targets = [], [], [], []

    def forward(self, x_spatial, x_scalar):
        return self.model(x_spatial, x_scalar)

    def _get_datamodule_attr(self, attr_name, default=None):
        if self.trainer and hasattr(self.trainer, 'datamodule') and self.trainer.datamodule and hasattr(self.trainer.datamodule, attr_name):
            return getattr(self.trainer.datamodule, attr_name)
        return default

    def on_fit_start(self):
        self.normalizer = self._get_datamodule_attr('normalizer', Normalizer())
        self.area_weights_xr = self._get_datamodule_attr('area_weights')
        self.output_climatology = self._get_datamodule_attr('output_climatology')
        self.pr_transformer = self._get_datamodule_attr('pr_transformer', PowerTransformer(method='yeo-johnson', standardize=False))
        self.dm_output_vars = self._get_datamodule_attr('output_vars', [])
        if self.area_weights_xr is None: print("CRITICAL WARNING: area_weights_xr is None in on_fit_start.")

    def _inverse_transform_batch(self, y_hat_norm_batch, batch_indices=None):
        if self.normalizer is None: self.normalizer = self._get_datamodule_attr('normalizer', Normalizer())
        if self.pr_transformer is None: self.pr_transformer = self._get_datamodule_attr('pr_transformer', PowerTransformer(method='yeo-johnson', standardize=False))
        current_output_vars = self.dm_output_vars if self.dm_output_vars else self._get_datamodule_attr('output_vars', [])
        use_clim_removal = self._get_datamodule_attr('use_climatology_removal', False)
        if self.output_climatology is None and use_clim_removal: self.output_climatology = self._get_datamodule_attr('output_climatology')

        y_hat_maybe_anom_np = self.normalizer.inverse_transform_output(y_hat_norm_batch.cpu().numpy())
        y_hat_physical_np = np.copy(y_hat_maybe_anom_np)

        if "pr" in current_output_vars and self.pr_transformer is not None and hasattr(self.pr_transformer, 'lambdas_'):
            try:
                pr_idx = current_output_vars.index("pr")
                n_s, _, n_y, n_x = y_hat_physical_np.shape
                pr_data_flat = y_hat_physical_np[:, pr_idx].reshape(-1, 1)
                if np.isnan(pr_data_flat).any(): pr_data_flat = np.nan_to_num(pr_data_flat, nan=0)
                y_hat_physical_np[:, pr_idx] = self.pr_transformer.inverse_transform(pr_data_flat).reshape(n_s, n_y, n_x)
            except Exception as e: print(f"Error during inverse Yeo-Johnson for 'pr' in batch: {e}")
        return y_hat_physical_np

    def compute_multi_component_loss(self, predictions, targets, phase="train"):
        losses = {}
        if torch.isnan(predictions).any(): print(f"NaN in loss predictions ({phase})")
        if torch.isnan(targets).any(): print(f"NaN in loss targets ({phase})")

        current_area_weights_xr = self.area_weights_xr if self.area_weights_xr is not None else self._get_datamodule_attr('area_weights')
        if current_area_weights_xr is None:
             lat_weights_tensor = torch.ones_like(predictions[0,0,:,0]).unsqueeze(0).unsqueeze(0).unsqueeze(-1).to(predictions.device)
        else:
            lat_weights_np = current_area_weights_xr.data
            lat_weights_tensor = torch.from_numpy(lat_weights_np).float().to(predictions.device)
            lat_weights_tensor = lat_weights_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        mse = (predictions - targets) ** 2
        weighted_mse = mse * lat_weights_tensor
        if torch.isnan(weighted_mse.mean()): print(f"NaN in weighted_mse.mean() for {phase} loss")

        losses['rmse'] = torch.sqrt(torch.mean(weighted_mse) + 1e-9)
        if torch.isnan(losses['rmse']): print(f"NaN in rmse loss ({phase})")

        if predictions.shape[0] >= 120 and predictions.dim() > 3 :
            n_decades = predictions.shape[0] // 120
            if n_decades > 0:
                pred_d = predictions[:n_decades*120].reshape(n_decades, 120, *predictions.shape[1:])
                target_d = targets[:n_decades*120].reshape(n_decades, 120, *targets.shape[1:])
                losses['decadal_mean'] = F.mse_loss(pred_d.mean(dim=1), target_d.mean(dim=1))
                losses['decadal_std'] = F.mse_loss(pred_d.std(dim=1), target_d.std(dim=1))
                if torch.isnan(losses.get('decadal_mean',torch.tensor(0.0))).any(): print(f"NaN in decadal_mean loss ({phase})")
                if torch.isnan(losses.get('decadal_std',torch.tensor(0.0))).any(): print(f"NaN in decadal_std loss ({phase})")

        if predictions.shape[1] > 1:
            pr_model_output_channel = predictions[:, 1:2]
            losses['pr_nonneg'] = torch.mean(torch.relu(-pr_model_output_channel))
            if torch.isnan(losses.get('pr_nonneg', torch.tensor(0.0))).any(): print(f"NaN in pr_nonneg loss ({phase})")
        else: losses['pr_nonneg'] = torch.tensor(0.0, device=predictions.device)

        total_loss_val = 0.0; valid_loss_components = 0
        for k, v_loss in losses.items():
            if torch.is_tensor(v_loss) and not torch.isnan(v_loss):
                total_loss_val += self.loss_weights.get(k, 0) * v_loss
                valid_loss_components +=1

        if valid_loss_components == 0 and losses:
             raise RuntimeError(f"All loss components NaN in {phase}. Stopping.")
        total_loss = total_loss_val if valid_loss_components > 0 else torch.tensor(0.0, device=predictions.device, requires_grad=True)

        for k, v_loss in losses.items():
            if torch.is_tensor(v_loss):
                log_val = v_loss.item() if not torch.isnan(v_loss) else float('nan')
                bs = self.trainer.datamodule.batch_size if self.trainer and self.trainer.datamodule else None
                self.log(f"{phase}/loss_{k}", log_val, on_step=(phase=="train"), on_epoch=True, prog_bar=False, logger=True, batch_size=bs)

        if torch.isnan(total_loss):
            raise RuntimeError(f"Total loss is NaN in {phase} phase. Losses: {losses}")
        return total_loss, losses

    def training_step(self, batch, batch_idx):
        x_spatial, x_scalar, y = batch
        if torch.isnan(x_spatial).any(): print(f"NaN in training_step x_spatial batch {batch_idx}")
        if torch.isnan(x_scalar).any(): print(f"NaN in training_step x_scalar batch {batch_idx}")
        if torch.isnan(y).any(): print(f"NaN in training_step y batch {batch_idx}")
        y_hat = self(x_spatial, x_scalar)
        if torch.isnan(y_hat).any(): print(f"NaN in training_step y_hat batch {batch_idx}")
        loss, _ = self.compute_multi_component_loss(y_hat, y, phase="train")
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch[0].size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x_spatial, x_scalar, y_norm_targets = batch
        y_hat_norm_space = self(x_spatial, x_scalar)
        loss, _ = self.compute_multi_component_loss(y_hat_norm_space, y_norm_targets, phase="val")
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch[0].size(0))

        y_hat_physical_no_clim = self._inverse_transform_batch(y_hat_norm_space.detach().clone())
        y_targets_physical_no_clim = self._inverse_transform_batch(y_norm_targets.detach().clone())

        self.val_preds.append(y_hat_physical_no_clim)
        self.val_targets.append(y_targets_physical_no_clim)
        return loss

    def on_validation_epoch_end(self):
        if not self.val_preds or not self.val_targets:
            print("Validation preds/targets list is empty. Skipping evaluation.")
            self.val_preds.clear(); self.val_targets.clear()
            return
        try:
            preds_np_no_clim = np.concatenate(self.val_preds, axis=0)
            trues_np_no_clim = np.concatenate(self.val_targets, axis=0)
        except ValueError as e:
            print(f"Error concatenating validation preds/targets: {e}.")
            self.val_preds.clear(); self.val_targets.clear()
            return

        preds_physical_final, trues_physical_final = np.copy(preds_np_no_clim), np.copy(trues_np_no_clim)
        current_output_climatology = self.output_climatology if self.output_climatology is not None else self._get_datamodule_attr('output_climatology')
        use_clim_removal = self._get_datamodule_attr('use_climatology_removal', False)
        n_train_samples = self._get_datamodule_attr('n_train_samples_for_clim_ref', 0)

        if current_output_climatology is not None and use_clim_removal:
            print("DEBUG: Adding climatology back to full validation predictions/targets.")
            start_month_val_idx = n_train_samples % 12
            for i in range(preds_physical_final.shape[0]):
                month_idx_0_11 = (start_month_val_idx + i) % 12
                preds_physical_final[i] += current_output_climatology[month_idx_0_11]
                trues_physical_final[i] += current_output_climatology[month_idx_0_11]

        if np.isnan(preds_physical_final).any(): print("NaNs found in final val_preds after climatology before _evaluate")
        if np.isnan(trues_physical_final).any(): print("NaNs found in final val_trues after climatology before _evaluate")

        self._evaluate(preds_physical_final, trues_physical_final, phase="val")
        self.val_preds.clear(); self.val_targets.clear()

    def test_step(self, batch, batch_idx):
        x_spatial, x_scalar, y_raw_targets_batch = batch
        y_hat_norm_space = self(x_spatial, x_scalar)
        y_hat_physical_no_clim = self._inverse_transform_batch(y_hat_norm_space.detach().clone())
        self.test_preds.append(y_hat_physical_no_clim)
        self.test_targets.append(y_raw_targets_batch.cpu().numpy())

    def on_test_epoch_end(self):
        if not self.test_preds or not self.test_targets:
            print("Test preds/targets list is empty. Skipping evaluation.")
            self.test_preds.clear(); self.test_targets.clear()
            return
        try:
            preds_np_no_clim = np.concatenate(self.test_preds, axis=0)
            trues_np_raw = np.concatenate(self.test_targets, axis=0)
        except ValueError as e:
            print(f"Error concatenating test preds/targets: {e}.")
            self.test_preds.clear(); self.test_targets.clear()
            return

        preds_physical_final = np.copy(preds_np_no_clim)
        current_output_climatology = self.output_climatology if self.output_climatology is not None else self._get_datamodule_attr('output_climatology')
        use_clim_removal = self._get_datamodule_attr('use_climatology_removal', False)

        if current_output_climatology is not None and use_clim_removal:
            print("DEBUG: Adding climatology back to full test predictions.")
            for i in range(preds_physical_final.shape[0]):
                month_idx_0_11 = i % 12
                preds_physical_final[i] += current_output_climatology[month_idx_0_11]

        if np.isnan(preds_physical_final).any(): print("NaNs found in final test_preds after climatology before _evaluate")
        if np.isnan(trues_np_raw).any(): print("NaNs found in final test_trues (raw) before _evaluate")

        self._evaluate(preds_physical_final, trues_np_raw, phase="test")
        self._save_submission(preds_physical_final)
        self.test_preds.clear(); self.test_targets.clear()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        max_epochs = self.trainer.max_epochs if self.trainer and self.trainer.max_epochs is not None else config["trainer"]["max_epochs"]
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def _evaluate(self, preds_physical, trues_physical, phase="val"):
        current_area_weights_xr = self.area_weights_xr if self.area_weights_xr is not None else self._get_datamodule_attr('area_weights')
        current_lat = self._get_datamodule_attr('lat', np.array([]))
        current_lon = self._get_datamodule_attr('lon', np.array([]))
        current_output_vars = self.dm_output_vars if self.dm_output_vars else self._get_datamodule_attr('output_vars', [])

        if current_area_weights_xr is None or not current_lat.size or not current_lon.size or not current_output_vars:
            print(f"Evaluation skipped for {phase}: Missing critical attributes.")
            return

        time_indices = np.arange(preds_physical.shape[0])
        if np.isnan(preds_physical).any(): print(f"NaNs in _evaluate preds_physical ({phase}) at start.")
        if np.isnan(trues_physical).any(): print(f"NaNs in _evaluate trues_physical ({phase}) at start.")

        for i, var_name in enumerate(current_output_vars):
            p_var, t_var = preds_physical[:, i], trues_physical[:, i]
            p_var_filled, t_var_filled = np.nan_to_num(p_var, nan=0.0), np.nan_to_num(t_var, nan=0.0)
            p_xr, t_xr = xr.DataArray(p_var_filled, dims=["time","y","x"], coords={"time":time_indices,"y":current_lat,"x":current_lon}), \
                         xr.DataArray(t_var_filled, dims=["time","y","x"], coords={"time":time_indices,"y":current_lat,"x":current_lon})

            rmse_val, mean_rmse_val, std_mae_val, acc_val = np.nan, np.nan, np.nan, np.nan
            try:
                rmse_val = np.sqrt(((p_xr - t_xr)**2).weighted(current_area_weights_xr).mean()).item()
                mean_rmse_val = np.sqrt(((p_xr.mean("time") - t_xr.mean("time"))**2).weighted(current_area_weights_xr).mean()).item()
                std_mae_val = np.abs(p_xr.std("time") - t_xr.std("time")).weighted(current_area_weights_xr).mean().item()
                p_anom, t_anom = p_xr - p_xr.mean("time"), t_xr - t_xr.mean("time")
                p_std_val, t_std_val = np.sqrt((p_anom**2).weighted(current_area_weights_xr).mean()), np.sqrt((t_anom**2).weighted(current_area_weights_xr).mean())

                if p_std_val.item() < 1e-9 or t_std_val.item() < 1e-9 or np.isnan(p_std_val.item()) or np.isnan(t_std_val.item()): acc_val = np.nan
                else: acc_val = ((p_anom * t_anom).weighted(current_area_weights_xr).mean() / (p_std_val * t_std_val + 1e-9)).mean().item()
            except Exception as e: print(f"Error calculating metrics for {var_name} ({phase}): {e}")

            print(f"[{phase.upper()}] {var_name}: RMSE={rmse_val:.4f}, Time-Mean RMSE={mean_rmse_val:.4f}, Time-Stddev MAE={std_mae_val:.4f}, ACC={acc_val:.4f}")
            bs = self.trainer.datamodule.batch_size if self.trainer and self.trainer.datamodule else None
            self.log_dict({
                f"{phase}/{var_name}/rmse":rmse_val, f"{phase}/{var_name}/time_mean_rmse":mean_rmse_val,
                f"{phase}/{var_name}/time_std_mae":std_mae_val, f"{phase}/{var_name}/acc":acc_val,
            }, logger=True, batch_size=bs)

    def _save_submission(self, predictions_physical):
        current_lat, current_lon = self._get_datamodule_attr('lat',np.array([])), self._get_datamodule_attr('lon',np.array([]))
        current_output_vars = self.dm_output_vars if self.dm_output_vars else self._get_datamodule_attr('output_vars',[])
        if not current_lat.size or not current_lon.size or not current_output_vars:
            print("Submission skipped: Missing lat, lon, or output_vars for submission.")
            return

        time_indices, rows = np.arange(predictions_physical.shape[0]), []
        filepath = f"submissions/enhanced_kaggle_submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        for t_idx, _ in enumerate(time_indices):
            for var_idx, var_name in enumerate(current_output_vars):
                for y_idx, y_val in enumerate(current_lat):
                    for x_idx, x_val in enumerate(current_lon):
                        pred_val = predictions_physical[t_idx,var_idx,y_idx,x_idx]
                        rows.append({"ID":f"t{t_idx:03d}_{var_name}_{y_val:.2f}_{x_val:.2f}", "Prediction":0.0 if np.isnan(pred_val) else pred_val})
        df = pd.DataFrame(rows); os.makedirs("submissions", exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"✅ Submission saved to: {filepath}")

def train_enhanced_model():
    print("--- DEBUG: Initializing DataModule ---")
    datamodule = EnhancedClimateDataModule(**config["data"])
    print("--- DEBUG: Setting up DataModule ---")
    datamodule.setup()

    try:
        temp_train_loader = datamodule.train_dataloader()
        if len(temp_train_loader) > 0:
            sample_batch = next(iter(temp_train_loader))
            n_spatial_channels = sample_batch[0].shape[1]
            n_scalar_inputs = sample_batch[1].shape[1]
        else:
            print("Train dataloader is empty. Using estimates for channel numbers.")
            n_spatial_channels = sum(1 for v in config["data"]["input_vars"] if v not in ["CO2", "SO2", "CH4", "BC"])
            n_scalar_inputs = len(config["data"]["input_vars"]) - n_spatial_channels + 3
        print(f"Spatial channels: {n_spatial_channels}, Scalar inputs: {n_scalar_inputs}")
    except Exception as e:
        print(f"Error getting sample batch: {e}. Using estimates.")
        n_spatial_channels = sum(1 for v in config["data"]["input_vars"] if v not in ["CO2", "SO2", "CH4", "BC"])
        n_scalar_inputs = len(config["data"]["input_vars"]) - n_spatial_channels + 3
        print(f"Estimated spatial channels: {n_spatial_channels}, Estimated scalar inputs: {n_scalar_inputs}")

    print("--- DEBUG: Initializing Model ---")
    model_config_params = {k:v for k,v in config["model"].items() if k != "type"}
    model = EnhancedUNetWithFiLM(
        n_input_channels=n_spatial_channels, n_output_channels=len(config["data"]["output_vars"]),
        n_scalar_inputs=n_scalar_inputs, **model_config_params
    )
    print("--- DEBUG: Initializing LightningModule ---")
    lm_params = {k: config["training"][k] for k in ["lr", "weight_decay", "loss_weights"] if k in config["training"]}
    lightning_module = EnhancedClimateEmulationModule(model, **lm_params)

    callbacks = [
        pl.callbacks.ModelCheckpoint(monitor="val/loss", dirpath="checkpoints", filename="best_model-{epoch:02d}-{val/loss:.4f}", save_top_k=1, mode="min", save_last=True),
        pl.callbacks.EarlyStopping(monitor="val/loss", patience=10, mode="min", verbose=True),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    ]
    trainer_params = {k:v for k,v in config["trainer"].items() if k != "callbacks"}

    print("--- DEBUG: Initializing Trainer ---")
    trainer = pl.Trainer(**trainer_params, callbacks=callbacks)
    print("--- DEBUG: Starting Training (trainer.fit) ---")
    trainer.fit(lightning_module, datamodule=datamodule)
    print("--- DEBUG: Starting Testing (trainer.test) ---")
    trainer.test(lightning_module, datamodule=datamodule, ckpt_path="best")
    return lightning_module, datamodule

if __name__ == "__main__":
    print("Enhanced Climate Emulation Model Training")
    print("="*60)
    try:
        model_module, datamodule_used = train_enhanced_model()
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    print("\nTraining complete!")
    print("="*60)
