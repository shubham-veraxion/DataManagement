def transform(df_dict):
    target_dataset = "NORDPOOL_CET_Spot_Power_EOD_Validated_1330_20260422133012"
    
    # Check if the target dataset exists in the dictionary
    if target_dataset not in df_dict:
        raise ValueError(f"Dataset '{target_dataset}' not found in the provided dataframes.")
    
    # Extract the target dataframe
    df = df_dict[target_dataset]
    
    # Validate column existence
    required_columns = ['commodity']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing in the dataset.")
    
    # Remove rows where commodity value is 'Block'
    transformed_df = df[df['commodity'] != 'Block']
    
    return transformed_df