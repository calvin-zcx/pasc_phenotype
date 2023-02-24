mkdir log_recover
python screen_dx_recover.py --site all --severity inpatient --negative_ratio 1 --downsample_ratio 1 2>&1 | tee  log_recover/screen_dx_recover_all_inpatient_neg1_downsample1.txt
python screen_dx_recover.py --site all --severity icu --negative_ratio 1 --downsample_ratio 1 2>&1 | tee  log_recover/screen_dx_recover_all_icu_neg1_downsample1.txt
python screen_dx_recover.py --site all --severity deltaAndBefore --negative_ratio 1 --downsample_ratio 1 2>&1 | tee  log_recover/screen_dx_recover_all_deltaAndBefore_neg1_downsample1.txt
python screen_dx_recover.py --site all --severity omicron --negative_ratio 1 --downsample_ratio 1 2>&1 | tee  log_recover/screen_dx_recover_all_omicron_neg1_downsample1.txt
