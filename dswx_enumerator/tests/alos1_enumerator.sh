python -m dswx_enumerator enumerate \
  --concept-id C1206487504-ASF \
  --db /mnt/aurora-r0/jungkyo/OPERA/DSWx-NI/R21_beta_patch/sample_product/input_dir/ancillary_data/MGRS_collection_db_DSWx-NI_v0.1.sqlite \
  --mgrs-set-id MS_38_66 \
  --start 2009-01-01T00:00:00Z \
  --end   2009-12-31T23:59:59Z \
  --track 20169 \
  --outdir data/ALOS \
  --yaml-out runconfigs/