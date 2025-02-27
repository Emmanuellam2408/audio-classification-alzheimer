import pandas as pd

def fuse_and_clean_data(mfcc, text_features, groundtruth):
    print("Fusing all the features in a finalized dataframe...")
    # final dataframe preparation
    mfcc = mfcc.sort_values('File')
    mfcc['File'] = mfcc['File'].str.replace('.mp3', '')
    mfcc = mfcc.drop('dx', axis=1)
    text_features['filename'] = text_features['filename'].str.replace('.mp3', '')
    #text_features = text_features.drop('dx', axis=1)

    final_features = mfcc.merge(groundtruth, left_on='File', right_on='adressfname')
    final_features = final_features.merge(text_features, left_on='File', right_on='filename')
    final_features = final_features.drop('adressfname', axis=1)
    final_features = final_features.drop('filename', axis=1)

    final_features.to_csv('final_important_features_10_15_sec.csv', index=False)


cwd = os.getcwd()
file_path_1 = os.path.join(cwd, 'au_cla_alz_master', 'mfcc_15_sec.csv')
file_path_2 = os.path.join(cwd, 'au_cla_alz_master', 'top_10_mfcc.csv')
file_path_3 = os.path.join(cwd, 'au_cla_alz_master', 'text_features.csv')
#file_path_4 = os.path.join(cwd, 'au_cla_alz_master', 'cleaned_training_groundtruth.csv')



mfcc_all = pd.read_csv(file_path_1)
mfcc_top = pd.read_csv(file_path_2)
text_features = pd.read_csv(file_path_3)
#training_groundtruth = pd.read_csv("file_path_4")

fuse_and_clean_data(mfcc_top, text_features) #, training_groundtruth)