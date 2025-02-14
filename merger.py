import pandas as pd
import os

def loop_through_folders(fp1, fp2):
    for item in sorted(os.listdir(fp1)):
        item_path1 = os.path.join(fp1, item)
        if os.path.isfile(item_path1):
            for item2 in sorted(os.listdir(fp2)):
                item_path2 = os.path.join(fp2, item2)
                if os.path.isfile(item_path2) and item == item2:
                    df1 = pd.read_csv(item_path1)
                    df2 = pd.read_csv(item_path2)
                    df_combined = pd.concat([df1, df2], ignore_index=True)
                    output_folder = "./combined_files"
                    os.makedirs(output_folder, exist_ok=True)
                    output_path = os.path.join(output_folder, item + '_combined_file.csv')
                    df_combined.to_csv(output_path, index=False)
                    break

# Example usage:
folder_path1 = "./Mar-Apr" # Replace with the actual path
folder_path2 = "./Apr-May" # Replace with the actual path
loop_through_folders(folder_path1, folder_path2)




# df1 = pd.read_csv('file1.csv')

# df2 = pd.read_csv('file2.csv')

# df_combined = pd.concat([df1, df2], ignore_index=True)

# df_combined.to_csv('combined_file.csv', index=False)