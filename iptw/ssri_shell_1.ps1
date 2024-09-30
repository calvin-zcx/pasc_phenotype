python screen_ssri_iptw_pcornet.py  --exptype ssri-base-180-0-clean  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-base-180-0-clean-mentalcov.txt
python screen_ssri_iptw_pcornet.py  --exptype ssri-acute0-15-clean  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-acute0-15-clean-mentalcov.txt
python screen_ssri_iptw_pcornet.py  --exptype ssriVSbupropion-base-180-0-clean  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssriVSbupropion-base-180-0-clean-mentalcov.txt
python screen_ssri_iptw_pcornet.py  --exptype ssriVSbupropion-acute0-15-clean  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssriVSbupropion-acute0-15-clean-mentalcov.txt

python screen_ssri_iptw_pcornet.py  --exptype ssri-base-180-0  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-base-180-0-mentalcov.txt
python screen_ssri_iptw_pcornet.py  --exptype ssri-acute0-15  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-acute0-15-mentalcov.txt
python screen_ssri_iptw_pcornet.py  --exptype ssriVSbupropion-base-180-0  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssriVSbupropion-base-180-0-mentalcov.txt
python screen_ssri_iptw_pcornet.py  --exptype ssriVSbupropion-acute0-15  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssriVSbupropion-acute0-15-mentalcov.txt



python screen_ssri_iptw_pcornet.py  --exptype ssri-post30  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-post30-mentalcov.txt
python screen_ssri_iptw_pcornet.py  --exptype ssri-post30-basemental  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-post30-basemental-mentalcov.txt
python screen_ssri_iptw_pcornet.py  --exptype ssri-post30-nobasemental  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-post30-nobasemental-mentalcov.txt

python screen_ssri_iptw_pcornet.py  --exptype ssri-acute0-15-incident  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-acute0-15-incident-mentalcov.txt

python screen_ssri_iptw_pcornet.py  --exptype ssri-base180-acutevsnot  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-base180-acutevsnot-mentalcov.txt
python screen_ssri_iptw_pcornet.py  --exptype ssri-base180withmental-acutevsnot  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-base180withmental-acutevsnot-mentalcov.txt
python screen_ssri_iptw_pcornet.py  --exptype ssri-acute0-15-incident_nobasemental 2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-acute0-15-incident_nobasemental-mentalcov.txt
python screen_ssri_iptw_pcornet.py  --exptype ssri-acute0-15-incident_norequiremental 2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-acute0-15-incident_norequiremental-mentalcov.txt


python screen_ssri_iptw_pcornet.py  --exptype ssri-acute0-15-incident-pax05 2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-ssri-acute0-15-incident-pax05-mentalcov.txt
python screen_ssri_iptw_pcornet.py  --exptype ssri-acute0-15-incident-continue 2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-ssri-acute0-15-incident-continue-mentalcov.txt
python screen_ssri_iptw_pcornet.py  --exptype ssri-acute0-15-incident-pax15 2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-ssri-acute0-15-incident-pax15-mentalcov.txt

python screen_ssri_iptw_pcornet.py  --exptype fluvoxamine-base180withmental-acutevsnot 2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-fluvoxamine-base180withmental-acutevsnot-mentalcov.txt
python screen_ssri_iptw_pcornet.py  --exptype fluvoxamine-base180-acutevsnot 2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-fluvoxamine-base180-acutevsnot-mentalcov.txt
python screen_ssri_iptw_pcornet.py  --exptype fluvoxamine-base180withmental-acutevsnot-continue 2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-fluvoxamine-base180withmental-acutevsnot-continue-mentalcov.txt


'fluvoxamine-base180withmental-acutevsnot',
'fluvoxamine-base180-acutevsnot',
'fluvoxamine-base180withmental-acutevsnot-continue',

##python screen_ssri_iptw_pcornet.py  --exptype ssri-acute0-7  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-acute0-7-mentalcov.txt
##python screen_ssri_iptw_pcornet.py  --exptype ssri-base-120-0  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-base-120-0-mentalcov.txt
##
##python screen_ssri_iptw_pcornet.py  --exptype snri-acute0-7  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-snri-acute0-7.txt
##python screen_ssri_iptw_pcornet.py  --exptype ssriVSsnri-base-180-0  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssriVSsnri-base-180-0.txt
##python screen_ssri_iptw_pcornet.py  --exptype ssriVSsnri-acute0-7  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssriVSsnri-acute0-7.txt
##
##python screen_ssri_iptw_pcornet.py  --exptype ssri-base-180-0 --severity '2022-03' 2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-base-180-0-sub202203.txt
##python screen_ssri_iptw_pcornet.py  --exptype ssri-acute0-7  --severity '2022-03' 2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-acute0-7-sub202203.txt
##python screen_ssri_iptw_pcornet.py  --exptype ssri-acute0-15  --severity '2022-03' 2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-acute0-15-sub202203.txt
##python screen_ssri_iptw_pcornet.py  --exptype ssri-base-120-0  --severity '2022-03' 2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-base-120-0-sub202203.txt

