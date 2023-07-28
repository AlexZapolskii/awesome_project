import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
import itertools
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from utils.utils import *
from utils.enums import *


pallete = plt.get_cmap("Set2")

def optimize(df_source: pd.DataFrame):
    """
    Прогоняет регрессию с заданынми параметрами
    оптимизации
    """

    # TODO: параметризация
    #       subnumbers
    #       по

    subnumbers = 5
    a_x = np.linspace(20, 50, 4)
    b_x = np.linspace(70, 99, 4)

    percentiles_border_combs = []  # x

    for r in itertools.product(a_x, b_x):
        if np.all(np.diff(r) >= 48):
            percentiles_border_combs.append(r)

    a_y = np.linspace(0.01, 0.51, 4)
    b_y = np.linspace(0.49, 0.99, 4)

    combs = []  # y
    for r in itertools.product(a_y, b_y):
        if (
                np.all(np.diff(r) <= 0.9)
                and np.all(np.diff(r) > 0)
                and np.all(np.diff(r) >= 0.45)
        ):
            combs.append(r)

    strength = np.linspace(0, 0.8, subnumbers)
    length = [2, 4, 6, 8, 10, 12]

    context_vars = [
        "stores",
        "seasonality",
        "competitors_list_tv",
        "new_covid",
        "sales_qsr",
        "dish_qnt_reg_negative",
        "average_price_dish_region_smooth_5",
        "price_lag_new_smooth_40",
        "dummy_apr"
    ]

    paid_vars_imp = [
        "gis_imp",
        "final_ooh",
        "final_tm",
        "reg_tv_imp",
        "full_yandex_maps_imp",
        "OOH_imp",
        "final_posm",
        "digital_2020_2022Q1_imp",
        "nat_tv_wo2020_product_imp_sov",
        "nat_tv_wo2020_vfm_imp_sov",
        "final_ap",
        "digital_none_youtube_imp"
    ]

    for i in paid_vars_imp:
        if np.sum(df_source[i]) == 0:
            print(i)
            df_source = df_source.drop(i, axis=1)
            paid_vars_imp.remove(i)

    for var in paid_vars_imp:

        print(f'Current_var {var}')
        ans = pd.DataFrame(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            columns=[
                "coef",
                "percentile_values",
                "percentiles",
                "mape",
                "r2",
                "r2_adj",
                "p_value",
                "negative_signs",
                "wrong_roi",
                "wrong_adstock",
                f"{var}_roi",
                "percentile_1",
                "percentile_2",
                "max_value_adstock",
                "ban_roi",
                "ban_adstock",
            ],
        )

        for s in (strength):
            for l in (length):
                for i in (combs):
                    for percentiles_border in percentiles_border_combs:
                        df = df_source.copy()
                        df["competitors_list_tv"] = (
                            Carryover(strength=0.8, length=12)
                            .fit(np.array(df["competitors_list_tv"]).reshape(-1, 1))
                            .transform(
                                np.array(df["competitors_list_tv"]).reshape(-1, 1)
                            )
                        )

                        df[f"{var}_c"] = (
                            Carryover(strength=s, length=l)
                            .fit(np.array(df[var]).reshape(-1, 1))
                            .transform(np.array(df[var]).reshape(-1, 1))
                        )
                        index = range(
                            len(df[f"{var}_c"][df[f"{var}_c"] > 0].sort_values())
                        )
                        percentiles = [
                            int(np.percentile(np.array(index), i))
                            for i in percentiles_border
                        ]
                        x_data = [
                            df[f"{var}_c"][df[f"{var}_c"] > 0].sort_values()[p]
                            for p in percentiles
                        ]
                        # x_data = [np.max(df[f'{var}_c'][df[f'{var}_c']>0].sort_values()) * p for p in combs[0]]
                        max_value_adstock = np.max(
                            df[f"{var}_c"][df[f"{var}_c"] > 0].sort_values()
                        )

                        y_data = i

                        data = pd.DataFrame([x_data, y_data]).T
                        data.columns = ["x", "y"]

                        data.iloc[0, 1] = data.iloc[0, 1] + 1e-10
                        data.iloc[1, 1] = data.iloc[1, 1] + 1e-11
                        # data.iloc[2, 1] = data.iloc[2, 1] + 1e-12

                        data["y"] = data["y"] + 1e-3

                        data["z"] = np.log(1 / data["y"] - 1)

                        x0 = (
                            data["z"][0] * data["x"][1] - data["z"][1] * data["x"][0]
                        ) / (data["z"][0] - data["z"][1])

                        if x0 == np.inf or x0 == np.nan or x0 == 0:
                            x0 = x0 + 1e-10

                        alpha_1 = np.round(data["z"][0] / (x0 - data["x"][0]), 15)
                        alpha_2 = data["z"][1] / (x0 - data["x"][1])

                        alpha = alpha_1

                        df[f"{var}_trans"] = (
                            Saturation(x0, alpha)
                            .fit(np.array(df[f"{var}_c"]).reshape(-1, 1))
                            .transform(np.array(df[f"{var}_c"]).reshape(-1, 1))
                        )

                        model = sm.ols(
                            formula=formula([f"{var}_trans"] + context_vars), data=df
                        ).fit(method="pinv")

                        # display(model.summary())

                        TV_Digital_OOH_Geo = [
                            "reg_tv_trp",
                            "digital_2020_2022Q1_imp",
                            "digital_youtube_imp",
                            "gis_imp",
                            "full_yandex_maps_imp",
                        ]

                        paid_vars_spend_WO_ap_tm = [
                            i
                            for i in paid_vars_imp
                            if "final_ap" not in i and "final_tm" not in i
                        ]

                        coef = pd.DataFrame(model.params).T

                        if var == "gis_imp":
                            roi = {
                                var: np.sum(
                                    df[f"{var}_trans"] * coef[f"{var}_trans"].values
                                )
                                * np.mean(df["avg_check"])
                                / np.sum(df["gis_spend"])
                            }

                        elif var == "reg_tv_imp":
                            roi = {
                                var: np.sum(
                                    df[f"{var}_trans"] * coef[f"{var}_trans"].values
                                )
                                * np.mean(df["avg_check"])
                                / np.sum(df["reg_tv_spend"])
                            }

                        elif var == "OOH_imp":
                            roi = {
                                var: np.sum(
                                    df[f"{var}_trans"] * coef[f"{var}_trans"].values
                                )
                                * np.mean(df["avg_check"])
                                / np.sum(df["OOH_spend"])
                            }

                        elif var == "full_yandex_maps_imp":
                            roi = {
                                var: np.sum(
                                    df[f"{var}_trans"] * coef[f"{var}_trans"].values
                                )
                                * np.mean(df["avg_check"])
                                / np.sum(df["full_yandex_maps_spend"])
                            }

                        elif var == "nat_tv_wo2020_product_imp_sov":
                            roi = {
                                var: np.sum(
                                    df[f"{var}_trans"] * coef[f"{var}_trans"].values
                                )
                                * np.mean(df["avg_check"])
                                / np.sum(df["nat_tv_product_spend"])
                            }

                        elif var == "nat_tv_wo2020_vfm_imp_sov":
                            roi = {
                                var: np.sum(
                                    df[f"{var}_trans"] * coef[f"{var}_trans"].values
                                )
                                * np.mean(df["avg_check"])
                                / np.sum(df["nat_tv_vfm_spend"])
                            }

                        elif var == "digital_2020_2022Q1_imp":
                            roi = {
                                var: np.sum(
                                    df[f"{var}_trans"] * coef[f"{var}_trans"].values
                                )
                                * np.mean(df["avg_check"])
                                / np.sum(df["digital_2020_2022Q1_spend"])
                            }

                        elif var == "digital_none_youtube_imp":
                            roi = {
                                var: np.sum(
                                    df[f"{var}_trans"] * coef[f"{var}_trans"].values
                                )
                                * np.mean(df["avg_check"])
                                / np.sum(df["digital_none_youtube_spend"])
                            }

                        else:
                            roi = {
                                var: np.sum(
                                    df[f"{var}_trans"] * coef[f"{var}_trans"].values
                                )
                                * np.mean(df["avg_check"])
                                / np.sum(df[var])
                            }

                        roi = (
                            pd.DataFrame.from_dict(roi, orient="index")
                            .reset_index()
                            .fillna(0)
                        )
                        roi.columns = ["chanel", "roi"]

                        wrong_roi = np.where(
                            0.5 < roi["roi"].values < 5,
                            1,
                            np.where(
                                roi["roi"].values > 5,
                                roi["roi"].values / 5,
                                np.where(
                                    roi["roi"].values < 0.5, 0.5 / roi["roi"].values, 0
                                ),
                            ),
                        )[0]

                        ban_roi = np.where(roi["roi"].values > 10, 1, 0)[0]

                        wrong_adstock = np.where(s > 95, np.round(s / 0.95, 5), 1)

                        y_pred = model.predict(df[context_vars + [f"{var}_trans"]])
                        y_pred = y_pred.reset_index()[0]
                        y_pred = y_pred.fillna(np.mean(y_pred))
                        mape = mean_absolute_percentage_error(df["sales"], y_pred)
                        r2 = np.round(model.rsquared, 5)
                        r2_adj = np.round(model.rsquared_adj, 5)
                        p_values_df = (
                            pd.DataFrame(model.pvalues)
                            .reset_index()
                            .rename(columns={"index": "variables", 0: "p"})
                        )
                        p_value = np.round(
                            p_values_df[p_values_df["variables"] == f"{var}_trans"][
                                "p"
                            ].values[0],
                            5,
                        )
                        linear_coef = coef[f"{var}_trans"].values[0]

                        if var == "gis_imp":
                            ban_adstock = not (0 <= s <= 0.3)

                        elif var == "final_ooh":
                            ban_adstock = not (0 <= s <= 0.85)

                        elif var == "final_tm":
                            ban_adstock = not (0 <= s <= 0.85)

                        elif var == "reg_tv_imp":
                            ban_adstock = not (0.15 <= s <= 0.85)

                        elif var == "full_yandex_maps_imp":
                            ban_adstock = not (0 <= s <= 0.3)

                        elif var == "OOH_imp":
                            ban_adstock = not (0 <= s <= 0.7)

                        elif var == "final_posm":
                            ban_adstock = not (0 <= s <= 0.85)

                        elif var == "digital_2020_2022Q1_imp":
                            ban_adstock = not (0.15 <= s <= 0.85)

                        elif var == "nat_tv_wo2020_product_imp_sov":
                            ban_adstock = not (0.15 <= s <= 0.85)

                        elif var == "nat_tv_wo2020_vfm_imp_sov":
                            ban_adstock = not (0.15 <= s <= 0.85)

                        elif var == "final_ap":
                            ban_adstock = not (0 <= s <= 0.3)

                        elif var == "final_booking_promo":
                            ban_adstock = not (0 <= s <= 0.7)

                        elif var == "digital_none_youtube_imp":
                            ban_adstock = not (0.15 <= s <= 0.85)

                        ans = ans.append(
                            pd.DataFrame(
                                [
                                    [
                                        linear_coef,
                                        x_data,
                                        y_data,
                                        mape,
                                        r2,
                                        r2_adj,
                                        p_value,
                                        f"model_{s}_{l}_{np.round(x0, 5)}_{alpha}",
                                        np.sum(coef[f"{var}_trans"] < 0),
                                        wrong_roi,
                                        wrong_adstock,
                                        np.round(roi["roi"].values[0], 2),
                                        y_data[0],
                                        y_data[1],
                                        max_value_adstock,
                                        ban_roi,
                                        ban_adstock,
                                    ]
                                ],
                                columns=[
                                    "coef",
                                    "percentile_values",
                                    "percentiles",
                                    "mape",
                                    "r2",
                                    "r2_adj",
                                    "p_value",
                                    "model",
                                    "negative_signs",
                                    "wrong_roi",
                                    "wrong_adstock",
                                    roi["chanel"].values[0] + "_roi",
                                    "percentile_1",
                                    "percentile_2",
                                    "max_value_adstock",
                                    "ban_roi",
                                    "ban_adstock",
                                ],
                            )
                        )

        context_vars = []

        ans = ans[1:].fillna(0)
        ans["mape_norm"] = ans["mape"] / np.mean(ans["mape"])
        ans["r2_adj_norm"] = ans["r2_adj"] / np.mean(ans["r2_adj"])
        ans["p_value"] = ans["p_value"]  # / np.mean(ans['p_value'])

        ans["p_value_norm"] = (
            ans["p_value"] / (np.mean(ans["p_value"]) + 1e-10) + 1
        )  # ans['p_value'] / np.mean(ans['p_value'])

        ans["final_metric"] = (
            ans["r2_adj_norm"]
            / ans["p_value_norm"]
            / ans["mape_norm"]
            / ans["wrong_roi"]
            / ans["wrong_adstock"]
        )
        ans = ans.sort_values("final_metric", ascending=False)

        ans["strength"] = pd.to_numeric(
            ans["model"].apply(lambda x: x.split("_")[1:][0])
        )
        ans["length"] = pd.to_numeric(ans["model"].apply(lambda x: x.split("_")[1:][1]))
        ans["x0"] = pd.to_numeric(ans["model"].apply(lambda x: x.split("_")[1:][2]))
        ans["alpha"] = ans["model"].apply(lambda x: x.split("_")[1:][3])
        ans["percentile_y_1"] = ans["percentiles"].apply(lambda x: np.round(x[0], 5))
        ans["percentile_y_2"] = ans["percentiles"].apply(lambda x: np.round(x[1], 5))

        #ans.to_excel(f'{f"/data/interim/{var}_trans"}_res.xlsx', index=False)
        ans.to_excel(f'{f"data/processed/{var}_trans"}_res.xlsx', index=False)


