
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

st.set_page_config(layout="wide", page_title="Universal Bank AI Dashboard")

data = pd.read_csv("UniversalBank.csv")
data.columns = data.columns.str.strip()

st.title("Universal Bank Personal Loan Marketing Intelligence Dashboard")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
"Overview",
"Descriptive Analytics",
"Predictive Modeling",
"Marketing Insights",
"Prediction Tool"
])

with tab1:
    st.header("Dataset Overview")
    st.write("Dataset shape:", data.shape)
    st.dataframe(data.head())

    col1,col2,col3 = st.columns(3)

    col1.metric("Total Customers", len(data))
    col2.metric("Loan Acceptance Rate", f"{data['Personal Loan'].mean()*100:.2f}%")
    col3.metric("Avg Income", f"${data['Income'].mean():.0f}k")

with tab2:
    st.header("Customer Demographics")

    col1,col2 = st.columns(2)

    with col1:
        fig = px.histogram(data, x="Age", title="Age Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(data, x="Income", title="Income Distribution")
        st.plotly_chart(fig, use_container_width=True)

    col3,col4 = st.columns(2)

    with col3:
        fig = px.box(data, x="Education", y="Income", title="Income by Education")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.scatter(data, x="Income", y="CCAvg", color="Personal Loan",
                         title="Income vs Credit Card Spend")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

drop_cols = ["Personal Loan","ID","ZIPCode","ZIP Code"]
X = data.drop(columns=[c for c in drop_cols if c in data.columns])
y = data["Personal Loan"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

models = {
"Decision Tree":DecisionTreeClassifier(),
"Random Forest":RandomForestClassifier(),
"Gradient Boosting":GradientBoostingClassifier()
}

results=[]
roc_data={}
conf_mats={}

for name,model in models.items():
    model.fit(X_train,y_train)
    preds=model.predict(X_test)
    probs=model.predict_proba(X_test)[:,1]

    results.append({
    "Model":name,
    "Train Accuracy":model.score(X_train,y_train),
    "Test Accuracy":accuracy_score(y_test,preds),
    "Precision":precision_score(y_test,preds),
    "Recall":recall_score(y_test,preds),
    "F1 Score":f1_score(y_test,preds)
    })

    fpr,tpr,_=roc_curve(y_test,probs)
    roc_data[name]=(fpr,tpr,auc(fpr,tpr))
    conf_mats[name]=confusion_matrix(y_test,preds)

with tab3:
    st.header("Model Performance")
    st.dataframe(pd.DataFrame(results))

    st.subheader("ROC Curve Comparison")
    fig,ax = plt.subplots()

    for name,(fpr,tpr,roc_auc) in roc_data.items():
        ax.plot(fpr,tpr,label=f"{name} AUC={roc_auc:.2f}")

    ax.plot([0,1],[0,1],'--')
    ax.legend()
    st.pyplot(fig)

    st.subheader("Confusion Matrices")
    for name,cm in conf_mats.items():
        st.write(name)
        fig,ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

rf = RandomForestClassifier()
rf.fit(X,y)

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

with tab4:
    st.header("Marketing Insights")
    st.write("Feature importance shows which customer traits drive loan acceptance.")

    fig = px.bar(importances, title="Drivers of Loan Acceptance")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(data, x="Income", color="Personal Loan", barmode="overlay")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(data, x="Family", color="Personal Loan")
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header("Upload Customer Data for Prediction")

    uploaded = st.file_uploader("Upload CSV")

    if uploaded:
        new_data = pd.read_csv(uploaded)
        new_data.columns = new_data.columns.str.strip()

        new_data = new_data.drop(columns=["ID","ZIPCode","ZIP Code"], errors="ignore")
        new_data = new_data[X.columns]

        preds = rf.predict(new_data)
        probs = rf.predict_proba(new_data)[:,1]

        new_data["Predicted Personal Loan"] = preds
        new_data["Acceptance Probability"] = probs

        st.dataframe(new_data)

        csv = new_data.to_csv(index=False).encode()
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
