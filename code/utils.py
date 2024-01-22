import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def import_data():

    try:
        data = pd.read_csv(
            filepath_or_buffer='../assets/default_of_credit_card_clients.csv',
            sep=',',
        )

        return data

    except Exception as e:
        print(f'Errore in utils/import_data: {str(e)}')

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

    print(marriage_counts)
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

    custom_labels = ["April", "May", "June", "July", "August", "September",' ',' ']
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
    custom_labels = ["April", "May", "June", "July", "August", "September"]
    pay_state_sums.plot(kind='bar', ax=plt6, color='lightcoral')
    plt6.set_title("Total Pay Statement Amounts")
    plt6.set_xlabel("Pay Month")
    plt6.set_ylabel("Total Amount")
    plt6.set_xticklabels(custom_labels)
    plt6.tick_params(axis='x', rotation=45)

def create_age_graph(data,plt4):

    age_col = data["AGE"]

    plt4.hist(age_col,bins=20,edgecolor="black")
    plt4.set_title("Age Distribution")
    plt4.set_xlabel("Ages")
    plt4.set_ylabel("Count")

    pass

def show_graphs(data):

    print(data)

    fig,axes=plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(15, 20)
    )

    plt1, plt2, plt3, plt4, plt5, plt6, plt7,plt8 = axes.flatten()

    #Gender

    create_gender_bar_graph(data,plt1)
    create_education_sex_graph(data,plt2)
    create_marriage_graph(data,plt3)
    create_past_payement_category_graph(data,plt7)
    create_bill_state_graph(data,plt5)
    create_payement_amount_graph(data,plt6),
    create_age_graph(data,plt4)

    plt.tight_layout()
    plt.show()