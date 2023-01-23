mkdir log
taskset --cpu-list 4-7 python screen_dx_vtrim_spline.py --dataset oneflorida --site all --severity all 2>&1 | tee  log/screen_dx_oneflorida_all_all_Vtrim_spline.txt
taskset --cpu-list 4-7 python screen_dx_vtrim_nonl.py --dataset oneflorida --site all --severity all 2>&1 | tee  log/screen_dx_oneflorida_all_all_Vtrim_nonlinear.txt

