# project.py


import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pd.options.plotting.backend = 'plotly'

from IPython.display import display

# DSC 80 preferred styles
pio.templates["dsc80"] = go.layout.Template(
    layout=dict(
        margin=dict(l=30, r=30, t=30, b=30),
        autosize=True,
        width=600,
        height=400,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        title=dict(x=0.5, xanchor="center"),
    )
)
pio.templates.default = "simple_white+dsc80"
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def clean_loans(loans):
    loans_copy = loans.copy()
    loans_copy['issue_d'] = pd.to_datetime(loans_copy['issue_d'], format='%b-%Y')
    loans_copy['term'] = loans_copy['term'].str.strip('months').astype(int) #should use str.lower() for series, astype cast panda series/df
    loans_copy['emp_title'] = loans_copy['emp_title'].str.lower().str.strip() #apply to emp_title col
    loans_copy['emp_title'] = loans_copy['emp_title'].apply(lambda x: 'registered nurse' if x == 'rn' else x) #if title specifically rn
    loans_copy['term_end'] = pd.to_datetime(loans_copy['issue_d'] + loans_copy['term'].apply(lambda x: pd.DateOffset(months = x)))

    return loans_copy



# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------

def correlations(df, pairs):
    result = {}

    for col1, col2 in pairs:
        
        correlation = df[col1].corr(df[col2])
        
        result[f'r_{col1}_{col2}'] = correlation
    
    return pd.Series(result)



# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def create_boxplot(loans):
    loans_c= loans.copy()
    bins = [580, 670, 740, 800, 850]
    labels = ["[580, 670)", "[670, 740)", "[740, 800)", "[800, 850)"]

    loans_c['Credit Score Range'] = pd.cut(loans_c['fico_range_low'], bins=bins, labels=labels, right=False)
    loans_c['Credit Score Range'] = pd.Categorical(loans_c['Credit Score Range'], categories=labels, ordered=True)

    loans_c['term'] = loans_c['term'].astype(str)

    pixels = px.box(
        loans_c,
        x='Credit Score Range',
        y='int_rate',
        color='term',  
        title='Interest Rate vs. Credit Score',
        labels={
            'int_rate': 'Interest Rate (%)',
            'term': 'Loan Length (Months)',
            'Credit Score Range': 'Credit Score Range'
        },
        category_orders={"Credit Score Range": labels,  "term": [ "36", "60"]},  
        color_discrete_map={'36': 'yellow', '60': 'blue'}  
    )

    return pixels


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def ps_test(loans, N):
    with_statement = loans[loans['desc'].notna()]['int_rate']
    without_statement = loans[loans['desc'].isna()]['int_rate']
    
    observed_stat = with_statement.mean() - without_statement.mean()
    
    null_stats = []
    for _ in range(N):
        
        permuted = np.random.permutation(loans['int_rate'].values)
        
        perm_with = permuted[:len(with_statement)]
        perm_without = permuted[len(with_statement):]
        
        null_stats.append(perm_with.mean() - perm_without.mean())
    
    null_stats = np.array(null_stats)
    p_value = (null_stats >= observed_stat).mean()
    
    return p_value
    
def missingness_mechanism():
    return 2
    
def argument_for_nmar():
    '''
    Put your justification here in this multi-line string.
    Make sure to return your string!
    '''
    out = '''Some personal statements may be missing not at random 
    if applicants with higher risk profiles are less likely to include them, 
    since applications with higher risk might not want to include their profiles because they don't want to get caught'''
    return out


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def tax_owed(income, brackets):
    tax = 0.0 
    for i in range(len(brackets)):
        
        rate, lower_limit = brackets[i]
    
        if i == len(brackets) - 1:
            if income > lower_limit:
                tax += (income - lower_limit) * rate
        else:
            upper_rate, upper_limit = brackets[i + 1]
            if income > lower_limit:
                taxable_income = min(income, upper_limit) - lower_limit
                tax += taxable_income * rate
    
    return tax


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def clean_state_taxes(state_taxes_raw): 
    state_taxes_cleaned = state_taxes_raw.dropna(how='all').copy() #rows completely null

    state_taxes_cleaned['State'] = state_taxes_cleaned['State'].apply(
        lambda x: x if isinstance(x, str) and x[0].isalpha() else None
    ).ffill()

    state_taxes_cleaned['Rate'] = state_taxes_cleaned['Rate'].str.replace('%', '').replace('none', '0').fillna(0).astype(float)
    
    state_taxes_cleaned['Rate'] = pd.to_numeric(state_taxes_cleaned['Rate']) / 100
    
    state_taxes_cleaned['Rate'] = state_taxes_cleaned['Rate'].round(2)
    
    state_taxes_cleaned['Lower Limit'] = state_taxes_cleaned['Lower Limit'].replace('[\$,]', '', regex=True)
    state_taxes_cleaned['Lower Limit'] = pd.to_numeric(state_taxes_cleaned['Lower Limit'], errors='coerce').fillna(0).astype(int)
    
    return state_taxes_cleaned


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def state_brackets(state_taxes):
    out = state_taxes.copy()
   
    out = out.groupby('State').apply(lambda x: list(zip(x['Rate'], x['Lower Limit']))).reset_index()
    out.columns = ['State','bracket_list']

    return out.set_index('State')

    
def combine_loans_and_state_taxes(loans, state_taxes):
    # Start by loading in the JSON file.
    # state_mapping is a dictionary; use it!
    import json
    state_mapping_path = Path('data') / 'state_mapping.json'
    with open(state_mapping_path, 'r') as f:
        state_mapping = json.load(f)
        
    # Now it's your turn:
    loans_copyy = loans.copy()
    gf = state_taxes.copy()
    gf['State'] = gf['State'].apply(lambda x: state_mapping[x])
    new_data = state_brackets(gf)
    out = loans_copyy.merge(new_data, left_on = 'addr_state', right_index = True)
    
    out = out.rename(columns = {'addr_state': 'State'}) 
    
    return out


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def find_disposable_income(loans_with_state_taxes):
    FEDERAL_BRACKETS = [
     (0.1, 0), 
     (0.12, 11000), 
     (0.22, 44725), 
     (0.24, 95375), 
     (0.32, 182100),
     (0.35, 231251),
     (0.37, 578125)
    ]
    copy_loans = loans_with_state_taxes.copy()
    def calculate_tax(income, brackets):
        tax = 0 
        for i in range(len(brackets)): #for multuple tup
            #take out orig brackets
            rate, lower_limit = brackets[i]
            if income > lower_limit:
                #check if income falls the last bracket, then check if income is in the current bracket
                if i == len(brackets) - 1 or income < brackets[i + 1][1]:
                    tax += rate * (income - lower_limit)
                    break
                    
                else:
                    tax += rate * (brackets[i + 1][1] - lower_limit)
                
        return tax

    copy_loans['federal_tax_owed'] = copy_loans['annual_inc'].apply(
        lambda income: calculate_tax(income, FEDERAL_BRACKETS)
    )
    
    copy_loans['state_tax_owed'] = copy_loans.apply(
        lambda row: calculate_tax(row['annual_inc'], row['bracket_list']), axis=1
    )
    
    copy_loans['disposable_income'] = (
        copy_loans['annual_inc'] - 
        copy_loans['federal_tax_owed'] - 
        copy_loans['state_tax_owed']
    )
    
    return copy_loans


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def aggregate_and_combine(loans, keywords, quantitative_column, categorical_column):
    result_dict = {}
    
    for keyword in keywords:
        
        keyword_df = loans[loans['emp_title'].str.contains(keyword, case=False)]
        
        grouped_means = round(keyword_df.groupby(categorical_column)[quantitative_column].mean(), 2)
        
        overall_mean = round(keyword_df[quantitative_column].mean(), 2)
        
        grouped_means['Overall'] = overall_mean

        result = keyword + "_mean_" + quantitative_column
        
        result_dict[result] = grouped_means
    
    result_df = pd.DataFrame(result_dict)
    
    return result_df


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def exists_paradox(loans, keywords, quantitative_column, categorical_column):
    keywords = [k.lower() for k in keywords]
    
    group1 = loans[loans['emp_title'].str.contains(keywords[0], case= False)]
    group2 = loans[loans['emp_title'].str.contains(keywords[1], case= False)]

    overall_mean_group1 = group1[quantitative_column].mean()
    overall_mean_group2 = group2[quantitative_column].mean()
    
    category_means_group1 = group1.groupby(categorical_column)[quantitative_column].mean()
    category_means_group2 = group2.groupby(categorical_column)[quantitative_column].mean()

    #check if contains paradox
    paradox_found = False
    
    for category in category_means_group1.index:
        
        if category in category_means_group2.index:
            
            if (overall_mean_group1 > overall_mean_group2 and category_means_group1[category] < category_means_group2[category]) or \
               (overall_mean_group1 < overall_mean_group2 and category_means_group1[category] > category_means_group2[category]):
                paradox_found = True
                
                break
    return paradox_found
    
def paradox_example(loans):
    return {
        'loans': loans,
        'keywords': ['teacher', 'manager'], 
        'quantitative_column': 'int_rate', 
        'categorical_column': 'verification_status' 
    }
