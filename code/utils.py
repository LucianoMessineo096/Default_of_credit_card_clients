import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import StandardScaler

#-----------------------------------PREPROCESSING---------------------------------------------------------------#

def show_heatmap(df):

    correlation = df.corr()

    plt.figure(figsize=(20,10))
    sns.heatmap(correlation,annot=True,vmin=-1,cmap='mako')
    plt.show()

def one_hot_encoding(df,dict):

    df = df.copy()

    for column, prefix in dict.items():

        dummies = pd.get_dummies(df[column], prefix=prefix)
        df = pd.concat([df,dummies],axis=1)
        df = df.drop(column,axis=1)

    return df 

def preprocessing(df):

    df = df.copy()

    #drop ID column 
    df =df.drop('ID',axis=1)

    #show features correlation
    show_heatmap(df)

    #hot encoding
    dict = {
        'EDUCATION' : 'EDU',
        'MARRIAGE' : 'MAR'
    }
    
    df = one_hot_encoding(df,dict)

    y = df['default payment next month'].copy()
    X = df.drop('default payment next month',axis=1).copy()

    #scaling
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X),columns= X.columns)


    return X, y


#-------------------------------------------------------------------------------------------------#

def import_data():

    try:
        data = pd.read_csv(
            filepath_or_buffer='../assets/default_of_credit_card_clients.csv'
        )

        return data

    except Exception as e:
        print(f'Errore in utils/import_data: {str(e)}')


def get_tensors(X_train,X_test,y_train,y_test):

    X_train_tensor = torch.tensor(X_train.values,dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values,dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values,dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values,dtype=torch.float32)

    return X_train_tensor,X_test_tensor,y_train_tensor,y_test_tensor

##--------------------------GRAPH----------------------------------------------## 
def create_gender_bar_graph(data, plt1):

    sex_col = data.loc[:, 'SEX']
    sex_counts = sex_col.value_counts(normalize=True) * 100

    colors = plt.cm.Paired(range(len(sex_counts)))

    plt1.pie(sex_counts, labels=["female","male"], autopct='%1.1f%%', colors=colors)
    plt1.set_title("Gender distribution")

def create_education_sex_graph(data, plt2):
    education_sex_counts = data.groupby(['EDUCATION', 'SEX']).size().unstack()

    male_counts = education_sex_counts[1].fillna(0)
    female_counts = education_sex_counts[2].fillna(0)

    plt2.bar(
        [
            "High_School_M", 
            "High_School_F",
            "University_M", 
            "University_F",
            "Graduate_School_M", 
            "Graduate_School_F",
            "Others_M",
            "Others_F"
        ],
        [
            male_counts[1], 
            female_counts[1],
            male_counts[2], 
            female_counts[2],
            male_counts[3], 
            female_counts[3],
            male_counts[4],
            female_counts[4]
        ],
        color=['#008080', '#008000'] * 4,
    )

    plt2.set_title("Education Distribution by Gender")
    plt2.set_xlabel("Education and Gender")
    plt2.set_ylabel("Count")
    plt2.tick_params(axis='x', rotation=45) 
    plt2.grid(axis='y', linestyle='--', alpha=0.7)   

def create_marriage_graph(data,plt3):

    marriage_col = data.loc[:,"MARRIAGE"]
    marriage_counts = marriage_col.value_counts()

    married , single , others, others2 = marriage_counts[1], marriage_counts[2], marriage_counts[3], marriage_counts[0]

    plt3.bar(
        ["married","single","others","others2"],
        [married,single,others,others2]
    )
    plt3.set_title("Marriage Distribution")
    plt3.set_xlabel("Marriage status")
    plt3.set_ylabel("Count")

def create_past_payement_category_graph(data,plt7):

    payment_history_columns = data.loc[:, 'PAY_0':'PAY_6']
    payment_history_counts = payment_history_columns.apply(lambda col: col.value_counts()).T

    custom_labels = ["September", "August", "July", "June", "May", "April",' ',' ']
    custom_labels_legend = [
        'paying duly',
        'paying duly',
        'paying in time',
        '1 month delay',
        '2 month delay',
        '3 month delay',
        '4 month delay',
        '5 month delay',
        '6 month delay',
        '7 month delay',
        '8 month delay',
        '9 month or above delay',

    ]

    #bar_width = 0.8 / len(custom_labels)

    payment_history_counts.plot(kind='bar', stacked=True, ax=plt7, xticks=range(len(custom_labels)))
    plt7.set_title("Past Payment History Categories Distribution")
    plt7.set_xlabel("Month")
    plt7.set_ylabel("Count")
    plt7.legend(custom_labels_legend,title="Past Payment Status", loc='upper right')
    plt7.set_xticklabels(custom_labels)
    plt7.tick_params(axis='x', rotation=45)
    plt7.grid(axis='y', linestyle='--', alpha=0.7)

def create_bill_state_graph(data,plt5):

    bill_state_columns = data.loc[:, 'BILL_AMT1':'BILL_AMT6']

    bill_state_sums = bill_state_columns.sum()
    custom_labels = ["September","August","July","June","May","April"]
    bill_state_sums.plot(kind='bar', ax=plt5, color='lightcoral')
    plt5.set_title("Total Bill Statement Amounts")
    plt5.set_xlabel("Billing Month")
    plt5.set_ylabel("Total Amount")
    plt5.set_xticklabels(custom_labels)
    plt5.tick_params(axis='x', rotation=45)
    
def create_payement_amount_graph(data,plt6):

    pay_state_columns = data.loc[:, 'PAY_AMT1':'PAY_AMT6']

    pay_state_sums = pay_state_columns.sum()
    custom_labels = ["September", "August", "July", "June", "May", "April"]
    pay_state_sums.plot(kind='bar', ax=plt6, color='lightcoral')
    plt6.set_title("Total Pay Statement Amounts")
    plt6.set_xlabel("Pay Month")
    plt6.set_ylabel("Total Amount")
    plt6.set_xticklabels(custom_labels)
    plt6.tick_params(axis='x', rotation=45)

def create_age_graph(data, plt4):
    age_col = data["AGE"]

    plt4.hist(age_col, bins=20, edgecolor="black")
    plt4.set_title("Age Distribution")
    plt4.set_xlabel("Ages")
    plt4.set_ylabel("Count")

def create_graduates_by_age_range_graph(data, plt9):
    graduates_data = data[data['EDUCATION'] == 2]

    age_ranges = [(20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80)]

    graduates_by_age_range = []
    for age_range in age_ranges:
        lower_bound, upper_bound = age_range
        num_graduates = graduates_data[(graduates_data['AGE'] >= lower_bound) & (graduates_data['AGE'] < upper_bound)].shape[0]
        graduates_by_age_range.append(num_graduates)

    plt9.bar(range(len(age_ranges)), graduates_by_age_range, align='center', color='skyblue')
    plt9.set_xticks(range(len(age_ranges)))
    plt9.set_xticklabels([f'{int(lower)}-{int(upper)}' for lower, upper in age_ranges])
    plt9.set_xlabel('Age Range')
    plt9.set_ylabel('Number of Graduates')
    plt9.set_title('Number of Graduates per Age Range')

def credit_limit_by_age(data):
    num_age_bins = 5
    num_limit_bins = 5

    age_bins = np.linspace(data['AGE'].min(), data['AGE'].max(), num_age_bins + 1)
    limit_bins = np.linspace(data['LIMIT_BAL'].min(), data['LIMIT_BAL'].max(), num_limit_bins + 1)

    grouped = data.groupby([pd.cut(data['AGE'], age_bins), pd.cut(data['LIMIT_BAL'], limit_bins)]).size().unstack()

    ax = grouped.plot(kind='bar', stacked=True, figsize=(10, 6))

    ax.set_xlabel('Credit Limit Ranges')
    ax.set_ylabel('Number of People')
    ax.set_title('Distribution of Credit Limit per Age Range')

    plt.xticks(np.arange(len(grouped.columns)), [f'{int(limit_bins[i])}-{int(limit_bins[i+1])}' for i in range(len(limit_bins) - 1)], rotation=45)
    plt.legend(title='Age Ranges')

def payment_status_pie_chart(data, plt10):
    num_paid = data[data['default payment next month'] == 0].shape[0]
    num_unpaid = data[data['default payment next month'] == 1].shape[0]

    total = num_paid + num_unpaid
    percent_paid = (num_paid / total) * 100
    percent_unpaid = (num_unpaid / total) * 100

    labels = ['Paid', 'Unpaid']
    sizes = [percent_paid, percent_unpaid]
    colors = ['#ff9999', '#66b3ff']
    explode = (0.1, 0)

    plt10.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt10.set_title('Percentage of Paid and Unpaid Customers')


def show_graphs(data):

    fig1,axes1=plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 10)
    )

    fig2,axes2=plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 10)
    )

    fig3,axes3=plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(10,10)
    )

    fig4, axes4 = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(10,10)
    )

    plt1,plt2 = axes1.flatten()
    plt3,plt4 = axes2.flatten()
    plt5,plt6,plt7,plt8 = axes3.flatten()
    plt9,plt10,plt11,plt12 = axes4.flatten()

    create_gender_bar_graph(data,plt1)
    create_education_sex_graph(data,plt2)
    create_marriage_graph(data,plt3)
    create_age_graph(data,plt4)
    create_bill_state_graph(data,plt5)
    create_payement_amount_graph(data,plt6),
    create_past_payement_category_graph(data,plt7)
    create_graduates_by_age_range_graph(data,plt9)
    credit_limit_by_age(data)
    payment_status_pie_chart(data,plt10)

    plt.tight_layout()
    plt.show()

##---------------------------------------------------------------------------##