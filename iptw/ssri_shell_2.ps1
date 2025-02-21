#python screen_ssri_iptw_pcornet.py  --exptype snri-base-180-0-clean  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-snri-base-180-0-clean-mentalcov.txt
#python screen_ssri_iptw_pcornet.py  --exptype snri-acute0-15-clean  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-snri-acute0-7-clean-mentalcov.txt
#python screen_ssri_iptw_pcornet.py  --exptype ssriVSsnri-base-180-0-clean  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssriVSsnri-base-180-0-clean-mentalcov.txt
#python screen_ssri_iptw_pcornet.py  --exptype ssriVSsnri-acute0-15-clean  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssriVSsnri-acute0-15-clean-mentalcov.txt
#
#
#python screen_ssri_iptw_pcornet.py  --exptype snri-base-180-0  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-snri-base-180-0-mentalcov.txt
#python screen_ssri_iptw_pcornet.py  --exptype snri-acute0-15  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-snri-acute0-7-mentalcov.txt
#python screen_ssri_iptw_pcornet.py  --exptype ssriVSsnri-base-180-0  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssriVSsnri-base-180-0-mentalcov.txt
#python screen_ssri_iptw_pcornet.py  --exptype ssriVSsnri-acute0-15  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssriVSsnri-acute0-15-mentalcov.txt


#python screen_ssri_iptw_pcornet.py  --exptype ssri-base180-acutevsnot --severity omicronbroad 2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-base180-acutevsnot-omicronbroad-mentalcov.txt
#python screen_ssri_iptw_pcornet.py  --exptype ssri-base180-acutevsnot --severity omicron 2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-base180-acutevsnot-omicron-mentalcov.txt
#python screen_ssri_iptw_pcornet.py  --exptype ssri-base180-acutevsnot --severity alpha 2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-base180-acutevsnot-alpha-mentalcov.txt
#python screen_ssri_iptw_pcornet.py  --exptype ssri-base180-acutevsnot --severity omicronafter 2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-base180-acutevsnot-omicronafter-mentalcov.txt
#

#python screen_ssri_iptw_pcornet.py  --exptype ssri-base180-acutevsnot --severity above65omicronbroad 2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-base180-acutevsnot-above65omicronbroad-mentalcov.txt
#python screen_ssri_iptw_pcornet.py  --exptype ssri-base180-acutevsnot --severity above65 2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-base180-acutevsnot-above65-mentalcov.txt
#python screen_ssri_iptw_pcornet.py  --exptype ssri-base180-acutevsnot --severity 35to50 2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-base180-acutevsnot-35to50-mentalcov.txt
#python screen_ssri_iptw_pcornet.py  --exptype ssri-base180-acutevsnot --severity 50to65 2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-base180-acutevsnot-50to65-mentalcov.txt


#python screen_ssri_iptw_pcornetV2.py  --exptype ssri-base180-acutevsnot --severity less50 2>&1 | tee  log_ssri_v2/screen_ssri_iptw_pcornet-ssri-base180-acutevsnot-less50-CFSCVDDeath.txt
#python screen_ssri_iptw_pcornetV2.py  --exptype ssri-base180-acutevsnot --severity above50 2>&1 | tee  log_ssri_v2/screen_ssri_iptw_pcornet-ssri-base180-acutevsnot-above50-CFSCVDDeath.txt
#python screen_ssri_iptw_pcornetV2.py  --exptype ssri-base180-acutevsnot --severity depression 2>&1 | tee  log_ssri_v2/screen_ssri_iptw_pcornet-ssri-base180-acutevsnot-depression-CFSCVDDeath.txt

python screen_ssri_iptw_pcornetV3.py  --exptype ssri-base180-acutevsnot --severity hispanic 2>&1 | tee  log_ssri_v3/screen_ssri_iptw_pcornetV3-ssri-base180-acutevsnot-hispanic-CFSCVDDeath.txt
python screen_ssri_iptw_pcornetV3.py  --exptype ssri-base180-acutevsnot --severity nonhispanic 2>&1 | tee  log_ssri_v3/screen_ssri_iptw_pcornetV3-ssri-base180-acutevsnot-nonhispanic-CFSCVDDeath.txt

python screen_ssri_iptw_pcornetV3.py  --exptype ssri-base180-acutevsnot --severity female 2>&1 | tee  log_ssri_v3/screen_ssri_iptw_pcornetV3-ssri-base180-acutevsnot-female-CFSCVDDeath.txt
python screen_ssri_iptw_pcornetV3.py  --exptype ssri-base180-acutevsnot --severity male 2>&1 | tee  log_ssri_v3/screen_ssri_iptw_pcornetV3-ssri-base180-acutevsnot-male-CFSCVDDeath.txt

python screen_ssri_iptw_pcornetV3.py  --exptype ssri-base180-acutevsnot --severity white 2>&1 | tee  log_ssri_v3/screen_ssri_iptw_pcornetV3-ssri-base180-acutevsnot-white-CFSCVDDeath.txt
python screen_ssri_iptw_pcornetV3.py  --exptype ssri-base180-acutevsnot --severity black 2>&1 | tee  log_ssri_v3/screen_ssri_iptw_pcornetV3-ssri-base180-acutevsnot-black-CFSCVDDeath.txt

python screen_ssri_iptw_pcornetV3.py  --exptype ssri-base180-acutevsnot --severity less50 2>&1 | tee  log_ssri_v3/screen_ssri_iptw_pcornetV3-ssri-base180-acutevsnot-less50-CFSCVDDeath.txt
python screen_ssri_iptw_pcornetV3.py  --exptype ssri-base180-acutevsnot --severity above50 2>&1 | tee  log_ssri_v3/screen_ssri_iptw_pcornetV3-ssri-base180-acutevsnot-above50-CFSCVDDeath.txt
python screen_ssri_iptw_pcornetV3.py  --exptype ssri-base180-acutevsnot --severity depression 2>&1 | tee  log_ssri_v3/screen_ssri_iptw_pcornetV3-ssri-base180-acutevsnot-depression-CFSCVDDeath.txt

python screen_ssri_iptw_pcornetV3.py  --exptype ssri-base180-acutevsnot --severity anxiety 2>&1 | tee  log_ssri_v3/screen_ssri_iptw_pcornetV3-ssri-base180-acutevsnot-anxiety-CFSCVDDeath.txt
python screen_ssri_iptw_pcornetV3.py  --exptype ssri-base180-acutevsnot --severity SMI 2>&1 | tee  log_ssri_v3/screen_ssri_iptw_pcornetV3-ssri-base180-acutevsnot-SMI-CFSCVDDeath.txt
python screen_ssri_iptw_pcornetV3.py  --exptype ssri-base180-acutevsnot --severity nonSMI 2>&1 | tee  log_ssri_v3/screen_ssri_iptw_pcornetV3-ssri-base180-acutevsnot-nonSMI-CFSCVDDeath.txt

python screen_ssri_iptw_pcornetV3.py  --exptype ssri-base180-acutevsnot --severity omicronbroad 2>&1 | tee  log_ssri_v3/screen_ssri_iptw_pcornetV3-ssri-base180-acutevsnot-omicronbroad-CFSCVDDeath.txt


##python screen_ssri_iptw_pcornet.py  --exptype snri-acute0-7  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-snri-acute0-7.txt
##python screen_ssri_iptw_pcornet.py  --exptype ssriVSsnri-acute0-7  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssriVSsnri-acute0-7.txt
##python screen_ssri_iptw_pcornet.py  --exptype snri-base-120-0  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-snri-base-120-0.txt
##python screen_ssri_iptw_pcornet.py  --exptype ssriVSsnri-base-120-0  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssriVSsnri-base-120-0.txt
##python screen_ssri_iptw_pcornet.py  --exptype ssri-acute0-15  --severity '2022-03' 2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-acute0-15-sub202203.txt
##python screen_ssri_iptw_pcornet.py  --exptype ssri-base-120-0  --severity '2022-03' 2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-base-120-0-sub202203.txt
##python screen_ssri_iptw_pcornet.py  --exptype ssriVSwellbutrin-acute0-15  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssriVSwellbutrin-acute0-15.txt
