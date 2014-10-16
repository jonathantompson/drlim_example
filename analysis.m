clearvars; clc; clear global; close all; format shortG; rng(1);

% Read in the input files
fid = fopen('results/x_low_dim.bin');
x_low_dim = double(fread(fid, '*single'));
fclose(fid);
fid = fopen('results/factors.bin');
factors = double(fread(fid, '*int32'));
fclose(fid);

dim = 3;

x_low_dim = reshape(x_low_dim, dim, length(x_low_dim) / dim)';
factors = reshape(factors, 3, length(factors) / 3)';

elevation_colors = squeeze(factors(:,2));
azimuth_colors = squeeze(factors(:,3));

if (dim == 2)
  figure;
  circle_size = 10;
  scatter(x_low_dim(:,1), x_low_dim(:,2), circle_size, ...
    elevation_colors);
  
  figure;
  scatter(x_low_dim(:,1), x_low_dim(:,2), circle_size, ...
    azimuth_colors');
elseif (dim == 3)
  figure;
  circle_size = 10;
  scatter3(x_low_dim(:,1), x_low_dim(:,2), x_low_dim(:,3), circle_size, ...
    elevation_colors, 'fill');
  
  figure;
  scatter3(x_low_dim(:,1), x_low_dim(:,2), x_low_dim(:,3), circle_size, ...
    azimuth_colors, 'fill');
else
  display('Cannot plot higher dims!')
end