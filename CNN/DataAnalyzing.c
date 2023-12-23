# Study the characteristics of the AD & CN populations (age, sex, MMS, cdr_global)
def characteristics_table(df, merged_df):
    """Creates a DataFrame that summarizes the characteristics of the DataFrame df"""
    diagnoses = np.unique(df.diagnosis.values)
    population_df = pd.DataFrame(index=diagnoses,
                                columns=['N', 'age', '%sexF', 'education',
                                         'MMS', 'CDR=0', 'CDR=0.5', 'CDR=1', 'CDR=2'])
    merged_df = merged_df.set_index(['participant_id', 'session_id'], drop=True)
    df = df.set_index(['participant_id', 'session_id'], drop=True)
    sub_merged_df = merged_df.loc[df.index]

    for diagnosis in population_df.index.values:
        diagnosis_df = sub_merged_df[df.diagnosis == diagnosis]
        population_df.loc[diagnosis, 'N'] = len(diagnosis_df)
        # Age
        mean_age = np.mean(diagnosis_df.age_bl)
        std_age = np.std(diagnosis_df.age_bl)
        population_df.loc[diagnosis, 'age'] = '%.1f ± %.1f' % (mean_age, std_age)
        # Sex
        population_df.loc[diagnosis, '%sexF'] = round((len(diagnosis_df[diagnosis_df.sex == 'F']) / len(diagnosis_df)) * 100, 1)
        # Education level
        mean_education_level = np.nanmean(diagnosis_df.education_level)
        std_education_level = np.nanstd(diagnosis_df.education_level)
        population_df.loc[diagnosis, 'education'] = '%.1f ± %.1f' % (mean_education_level, std_education_level)
        # MMS
        mean_MMS = np.mean(diagnosis_df.MMS)
        std_MMS = np.std(diagnosis_df.MMS)
        population_df.loc[diagnosis, 'MMS'] = '%.1f ± %.1f' % (mean_MMS, std_MMS)
        # CDR
        for value in ['0', '0.5', '1', '2']:
          population_df.loc[diagnosis, 'CDR=%s' % value] = len(diagnosis_df[diagnosis_df.cdr_global == float(value)])

    return population_df

population_df = characteristics_table(OASIS_df, OASIS_df)
population_df
