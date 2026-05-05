def transform(df_dict):
    # Extract the target dataset from the dictionary
    target_dataset = "NORDPOOL_CET_Spot_Power_EOD_Validated_1330_20260422133012"
    if target_dataset not in df_dict:
        raise ValueError(f"Dataset '{target_dataset}' not found in the provided dictionary.")
    
    # Extract the DataFrame
    df = df_dict[target_dataset]
    
    # Remove rows where price is null
    df_transformed = df.dropna(subset=['price'])
    
    return df_transformed