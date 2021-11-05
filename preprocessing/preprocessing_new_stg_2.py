import pandas as pd
from glob import glob
from os import chdir
from sklearn.preprocessing import StandardScaler

pd.set_option('display.precision', 3)
window_size = 20
price_sma_cols = ['p_sma1', 'p_sma2', 'p_sma3']
volume_sma_cols = ['v_sma1', 'v_sma2', 'v_sma3']
price_ema_cols = ['p_ema1', 'p_ema2', 'p_ema3']
volume_ema_cols = ['v_ema1', 'v_ema2', 'v_ema3']
price_cols = ['price_'+str(i+1) for i in range(1, window_size, 2)]
columns_to_remove = ['ticker'] + ['date'] + price_cols + price_sma_cols + price_ema_cols + volume_sma_cols + volume_ema_cols
iex_stg1_output_folder = '/Users/dkadur/Documents/day_trade_lstm-main/CSVs/preprocessing_27/'
iex_stg2_output_folder = '/Users/dkadur/Documents/day_trade_lstm-main/CSVs/preprocessing_28/'
output_file = 'final_28_2%_OUT.CSV'
chdir(iex_stg1_output_folder)
iex_stg1_files = ['final_27_2%_OUT.CSV']

def process_stg1_output_files():

    for iex_stg1_file in iex_stg1_files:
        all_data_df = pd.DataFrame()

        iex_stg1_df = pd.read_csv(iex_stg1_file)

        iex_stg1_df['price_diff_2_4'] = iex_stg1_df['price_4'] - iex_stg1_df['price_2']
        iex_stg1_df['price_diff_2_6'] = iex_stg1_df['price_6'] - iex_stg1_df['price_2']
        iex_stg1_df['price_diff_2_8'] = iex_stg1_df['price_8'] - iex_stg1_df['price_2']
        iex_stg1_df['price_diff_2_10'] = iex_stg1_df['price_10'] - iex_stg1_df['price_2']
        iex_stg1_df['price_diff_2_12'] = iex_stg1_df['price_12'] - iex_stg1_df['price_2']
        iex_stg1_df['price_diff_2_14'] = iex_stg1_df['price_14'] - iex_stg1_df['price_2']
        iex_stg1_df['price_diff_2_16'] = iex_stg1_df['price_16'] - iex_stg1_df['price_2']
        iex_stg1_df['price_diff_2_18'] = iex_stg1_df['price_18'] - iex_stg1_df['price_2']
        iex_stg1_df['price_diff_2_20'] = iex_stg1_df['price_20'] - iex_stg1_df['price_2']

        iex_stg1_df['p_sma_12'] = iex_stg1_df['p_sma1'] - iex_stg1_df['p_sma2']
        iex_stg1_df['p_sma_13'] = iex_stg1_df['p_sma1'] - iex_stg1_df['p_sma3']
        iex_stg1_df['p_sma_23'] = iex_stg1_df['p_sma2'] - iex_stg1_df['p_sma3']

        iex_stg1_df['p_ema_12'] = iex_stg1_df['p_ema1'] - iex_stg1_df['p_ema2']
        iex_stg1_df['p_ema_13'] = iex_stg1_df['p_ema1'] - iex_stg1_df['p_ema3']
        iex_stg1_df['p_ema_23'] = iex_stg1_df['p_ema2'] - iex_stg1_df['p_ema3']

        iex_stg1_df['v_sma_12'] = iex_stg1_df['v_sma1'] - iex_stg1_df['v_sma2']
        iex_stg1_df['v_sma_13'] = iex_stg1_df['v_sma1'] - iex_stg1_df['v_sma3']
        iex_stg1_df['v_sma_23'] = iex_stg1_df['v_sma2'] - iex_stg1_df['v_sma3']

        iex_stg1_df['v_ema_12'] = iex_stg1_df['v_ema1'] - iex_stg1_df['v_ema2']
        iex_stg1_df['v_ema_13'] = iex_stg1_df['v_ema1'] - iex_stg1_df['v_ema3']
        iex_stg1_df['v_ema_23'] = iex_stg1_df['v_ema2'] - iex_stg1_df['v_ema3']

        columns = list(iex_stg1_df.columns)
        columns_to_standardize = columns[26:]

        iex_stg1_df[columns_to_standardize] = StandardScaler().fit_transform(iex_stg1_df[columns_to_standardize])
        iex_stg1_df = iex_stg1_df.drop(columns=columns_to_remove)
        iex_stg1_df[columns_to_standardize] = iex_stg1_df[columns_to_standardize].round(5)
        reorder_columns = columns_to_standardize + ['target']
        iex_stg1_df = iex_stg1_df[reorder_columns]
        all_data_df = all_data_df.append(iex_stg1_df)

        all_data_df = all_data_df.reset_index(drop=True)
        iex_stg2_file = iex_stg2_output_folder + output_file
        all_data_df.to_csv(iex_stg2_file)


if __name__ == '__main__':
    process_stg1_output_files()