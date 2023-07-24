"""
Константы
"""

competitors_list_w_o_tv = [
    "mcd_radio_imp",
    "mcd_оон_imp",
    "mcd_digital_imp",
    "kfc_radio_imp",
    "kfc_оон_imp",
    "kfc_digital_imp"
]

competitors_list_tv = ["mcd_nattv_imp", "kfc_nattv_imp", "mcd_regtv_imp", "kfc_regtv_imp"]

digital_list = [
    "digital_imp_вконтакте",
    "digital_imp_инстаграм",
    "digital_imp_mytarget",
    "digital_imp_yandex",
    "digital_imp_other_olv",
    "digital_imp_qbid",
    "digital_imp_rutube",
    "digital_imp_мтс",
    "digital_imp_redllama",
    "digital_imp_buzzoola",
    "digital_imp_ozon",
    "digital_imp_tiktok",
    "digital_imp_ivi_smart",
    "digital_imp_da",
    "digital_imp_gpm",
    "digital_imp_plazkart",
    "digital_imp_yabbi",
    "digital_imp_admile",
    "digital_imp_youtube",
    "digital_imp_telegram"

]

digital_spend_list = [
    'digital_spend_вконтакте',
    'digital_spend_инстаграм', 'digital_spend_mytarget',
    'digital_spend_yandex', 'digital_spend_other_olv', 'digital_spend_qbid',
    'digital_spend_rutube', 'digital_spend_мтс', 'digital_spend_redllama',
    'digital_spend_buzzoola', 'digital_spend_ozon', 'digital_spend_ivi_smart', 'digital_spend_da',
    'digital_spend_tiktok',
    'digital_spend_gpm', "digital_spend_youtube", "digital_spend_telegram"
]

paid_vars_imp = ["gis_imp",
                 'final_ooh',
                 "final_tm",
                 "reg_tv_imp",
                 "full_yandex_maps_imp",
                 "OOH_imp",
                 "final_posm",
                 'digital_2020_2022Q1_imp',
                 "nat_tv_wo2020_product_imp_sov",
                 "nat_tv_wo2020_vfm_imp_sov",
                 "final_ap",
                 # "nat_tv_wo2020_angus_imp_norm_sov",
                 'digital_none_youtube_imp']

paid_vars_spend = ["gis_spend",
                   'final_ooh',
                   "final_tm",
                   "reg_tv_spend",
                   "full_yandex_maps_spend",
                   "OOH_spend",
                   "final_posm",
                   'digital_2020_2022Q1_spend',
                   "nat_tv_product_spend",
                   "nat_tv_vfm_spend",
                   "final_ap",
                   # "nat_tv_angus_spend",
                   'digital_none_youtube_spend']

context_vars = ['stores', 'seasonality', 'competitors_list_tv', 'new_covid',
                'sales_qsr',
                'dish_qnt_reg_negative',
                'average_price_dish_region_smooth_5', 'price_lag_new_smooth_40',
                'dummy_apr']

TV_Digital_OOH_Geo = ['reg_tv_imp',
                      'digital_none_youtube_imp',
                      'gis_imp', 'full_yandex_maps_imp', 'digital_2020_2022Q1_imp']


context_vars_south_23 = ['stores', 'seasonality', 'competitors_list_tv', 'new_covid',
       'sales_qsr',
        'dish_qnt_reg_negative',
       'average_price_dish_region_smooth_5', 'price_lag_new_smooth_40',
       'dummy_apr']
