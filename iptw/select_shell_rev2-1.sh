mkdir log
taskset --cpu-list 0-3 python screen_dx_vtrim_spline.py --dataset V15_COVID19 --site ALL --severity all 2>&1 | tee  log/screen_dx_insight_ALL_all_Vtrim_spline.txt
taskset --cpu-list 0-3 python screen_dx_vtrim_nonl.py --dataset V15_COVID19 --site ALL --severity all 2>&1 | tee  log/screen_dx_insight_ALL_all_Vtrim_nonlinear.txt
