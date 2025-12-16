import xarray as xr
import xgcm
import roms_tools
import xwmt

def bite_sized_chunks(ds):
    chunk_dict = {
        d:100 for d in ds.sizes.keys()
        if (d not in ["time"]) & ("s_" != d[:2])
    }
    return ds.chunk({**{"time":1}, **chunk_dict})

def assign_grid_coords(ds, og):
    
    ds = ds.assign_coords({v.name:v for v in ds.data_vars.values() if len(v.dims) < 3})
    ds = ds.assign_coords({k:v for (k,v) in og.data_vars.items()})
    ds = ds.rename({"sc_r": "sigma_r", "sc_w":"sigma_w"})

    # Add necessary grid metrics and coordinates
    ds = ds.assign_coords({
        'areacello': (1/ds.pm)*(1/ds.pn),
        'dx_rho': 1/ds.pm,
        'dy_rho': 1/ds.pn,
        'lon': ds.lon_rho,
        'lat': ds.lat_rho,
        'geolon_c': ds.lon_psi,
        'geolat_c': ds.lat_psi
    })

    ds = ds.isel({
        "xi_rho":slice(1, ds.xi_rho.size-1),
        "eta_rho":slice(1, ds.eta_rho.size-1),
        "eta_u":slice(1, ds.eta_u.size-1),
        "xi_v":slice(1, ds.xi_v.size-1),
    })

    return ds

def create_roms_grid(ds):
    coords = {
        'X': {'center':  'xi_rho', 'outer':  f'xi_psi'},
        'Y': {'center': 'eta_rho', 'outer': f'eta_psi'},
        'Z': {'center':   's_rho', 'outer':   's_w'  }
    }
    boundary = {'X':'extend', 'Y':'extend', 'Z':'extend'}
    grid = xgcm.Grid(
        ds,
        coords=coords,
        boundary=boundary,
        metrics={("X","Y"):["areacello"]},
        autoparse_metadata=False
    )

    grid._ds = grid._ds.assign_coords({
        'dy_u': grid.interp(grid._ds.dy_rho.chunk({'xi_rho':-1}), 'X'),
        'dx_v': grid.interp(grid._ds.dx_rho.chunk({'eta_rho':-1}), 'Y')
    })

    return grid

def infer_thkcello(ds):
    ds["z_l"] = roms_tools.vertical_coordinate.compute_depth_coordinates(
        ds,
        depth_type="layer",
        location="rho",
        zeta=ds.zeta
    )
    
    ds["z_i"] = roms_tools.vertical_coordinate.compute_depth_coordinates(
        ds,
        depth_type="interface",
        location="rho",
        zeta=ds.zeta
    )
    grid = create_roms_grid(ds)
    
    ds["thkcello"] = -grid.diff(ds.z_i, "Z", boundary="extend")

    return ds

def swap_redundant_dimensions(ds):
    coord_remapping = {
        "xi_u":"xi_psi",
        "eta_v":"eta_psi",
        "xi_v":"xi_rho",
        "eta_u":"eta_rho"
    }

    for k,v in {**ds.data_vars, **ds.coords}.items():
        for c, c_new in coord_remapping.items():
            if c in v.dims:
                ds[k] = ds[k].swap_dims({c:c_new})

    ds = ds.assign_coords({
        c: xr.DataArray(ds[c].values, dims=(c_new,))
        for c, c_new in coord_remapping.items()
        if c in ds.coords
    })

    ds = ds.drop_dims([c for c in coord_remapping.keys() if c in ds.dims])

    ds = ds.squeeze()

    return ds
    
def load_data():
    og = xr.open_dataset("rrexnums_grd.nc")
    
    ds_snap = xr.open_dataset("rrexnum200_his.00000.nc")
    ds_snap = assign_grid_coords(ds_snap, og)
    ds_snap = bite_sized_chunks(ds_snap)
    ds_snap = infer_thkcello(ds_snap)

    wm_snap = xwmt.WaterMass(
        create_roms_grid(ds_snap), 
        t_name='temp',
        s_name='salt'
    )
    ds_snap["sigma2"] = wm_snap.get_density("sigma2")

    ds_snap = ds_snap.rename({
        **{'time':'time_bounds'},
        **{v:f"{v}_bounds" for v in ds_snap.data_vars}
    })
    
    ds_avg = xr.open_dataset("rrexnum200_avg.00000.nc")
    ds_avg = assign_grid_coords(ds_avg, og)    
    ds_avg = bite_sized_chunks(ds_avg)

    ds_dia = xr.open_dataset("rrexnum200_diaT_avg.00000.nc")
    ds_dia = assign_grid_coords(ds_dia, og)
    ds_dia = bite_sized_chunks(ds_dia)

    ds = xr.merge([ds_avg, ds_dia])
    ds = infer_thkcello(ds)

    ds = xr.merge([ds, ds_snap], compat="override")

    wm = xwmt.WaterMass(
        create_roms_grid(ds), 
        t_name='temp',
        s_name='salt'
    )
    ds["sigma2"] = wm.get_density("sigma2")

    ds = swap_redundant_dimensions(ds)
    
    return ds