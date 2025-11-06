import streamlit as st
import pandas as pd
import numpy as np
import chardet
import matplotlib.pyplot as plt
import seaborn as sns
import re
import html
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from lime.lime_text import LimeTextExplainer
from wordcloud import WordCloud

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="Naive Bayes + LIME (Custom vs Inbuilt)", layout="wide")
st.title("LIME Explainable Naive Bayes — Custom vs Inbuilt")

# -------------------------------
# Custom Hardcoded MultinomialNB
# -------------------------------
class HardcodedMultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        n_classes = len(np.unique(y))
        n_features = X.shape[1]
        self.classes_ = np.unique(y)

        class_count = np.zeros(n_classes)
        feature_count = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            class_count[i] = X_c.shape[0]
            fc = X_c.sum(axis=0)
            feature_count[i, :] = np.asarray(fc).ravel()

        smoothed_fc = feature_count + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1).reshape(-1, 1)
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc)
        self.class_log_prior_ = np.log(class_count) - np.log(class_count.sum())

    def predict(self, X):
        jll = X @ self.feature_log_prob_.T + self.class_log_prior_
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_proba(self, X):
        jll = X @ self.feature_log_prob_.T + self.class_log_prior_
        log_prob = jll - jll.max(axis=1, keepdims=True)
        prob = np.exp(log_prob)
        return prob / prob.sum(axis=1, keepdims=True)


# -------------------------------
# Helper Functions
# -------------------------------
def plot_confusion_matrix(y_true, y_pred, model_name):
    plt.figure(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot(plt.gcf())
    plt.close()

def generate_wordcloud(text_data, title):
    text = " ".join(pd.Series(text_data).astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    st.pyplot(plt.gcf())
    plt.close()

def robust_highlight(sample_text: str, lime_features):
    weights = {}
    for w, s in lime_features:
        lw = w.lower()
        weights[lw] = s if (lw not in weights or abs(s) > abs(weights[lw])) else weights[lw]

    parts = re.findall(r"\w+|\W+", sample_text)
    out = []
    for p in parts:
        if re.match(r"\w+", p):
            key = p.lower()
            score = weights.get(key, 0.0)
            if score != 0.0:
                color = "palegreen" if score > 0 else "salmon"
                out.append(f"<mark style='background-color:{color}; padding:2px; border-radius:4px;'>{html.escape(p)}</mark>")
            else:
                out.append(html.escape(p))
        else:
            out.append(html.escape(p))
    return "".join(out)


def render_model_page(model_key, model_title):
    if not st.session_state.get("trained", False):
        st.info("👆 Upload data, choose columns, and click **Train Models** in the sidebar.")
        return

    accuracy = st.session_state[f"accuracy_{model_key}"]
    st.header(model_title)
    st.success(f"✅ {model_title} | **Accuracy:** {accuracy*100:.2f}%")
    st.write(f" **Test Samples:** {len(st.session_state.y_test)}")

    y_test = st.session_state.y_test
    y_pred = st.session_state[f"y_pred_{model_key}"]
    class_names = st.session_state.class_names

    st.subheader("📊 Model Performance")
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    st.text(report)
    plot_confusion_matrix(y_test, y_pred, model_title)

    with st.expander("Word Cloud of Training Text", expanded=True):
        generate_wordcloud(st.session_state.X_train_text, "Training Text WordCloud")

    # ------------------ LIME ------------------
    st.subheader(" LIME Explainability")
    X_test_text = st.session_state.X_test_text
    n_test = len(X_test_text)
    st.caption(f"Total Test Data Points: {n_test}")

    index = st.number_input("Select Test Index", 0, n_test - 1, 0, key=f"idx_{model_key}")
    num_features = st.slider("# LIME Features", 5, 20, 8, key=f"nf_{model_key}")
    top_labels = st.slider("# Top Labels", 1, 3, 1, key=f"tl_{model_key}")

    sample_text = X_test_text.iloc[index]
    model = st.session_state[f"model_{model_key}"]
    vectorizer = st.session_state[f"vectorizer_{model_key}"]

    explainer = LimeTextExplainer(class_names=class_names)

    def predict(texts):
        return model.predict_proba(vectorizer.transform(texts))

    explanation = explainer.explain_instance(sample_text, predict, num_features=num_features, top_labels=top_labels)

    predicted_label = model.predict(vectorizer.transform([sample_text]))[0]
    actual_label = y_test[index]
    st.info(f"**Prediction:** {class_names[predicted_label]} | **Actual:** {class_names[actual_label]}")

    st.write("#### 📍 Prediction Probabilities")
    probs = model.predict_proba(vectorizer.transform([sample_text]))[0]
    prob_df = pd.DataFrame({"Class": class_names, "Probability": probs}).sort_values(by="Probability", ascending=False)
    st.table(prob_df.style.format({"Probability": "{:.2%}"}))

    lime_features = explanation.as_list(label=predicted_label)
    highlighted_html = robust_highlight(sample_text, lime_features)

    st.write("### Highlighted Text Explanation")
    st.markdown(f"<div style='font-size:18px; line-height:1.6;'>{highlighted_html}</div>", unsafe_allow_html=True)

    st.write("### LIME Feature Importance")
    fig = explanation.as_pyplot_figure(label=predicted_label)
    st.pyplot(fig)
    plt.close()


def render_compare_page():
    if not st.session_state.get("trained", False):
        st.info("👆 Train models first using the sidebar.")
        return

    st.header("📊 Compare Models (Accuracy)")

    acc_custom = st.session_state["accuracy_custom"]
    acc_sk = st.session_state["accuracy_sk"]

    st.write(f"**Custom NB (TF-IDF):** {acc_custom*100:.2f}%")
    st.write(f"**Sklearn NB (CountVec 1,2):** {acc_sk*100:.2f}%")

    names = ["Custom NB (TF-IDF)", "Sklearn NB (Count 1,2)"]
    vals = [acc_custom, acc_sk]

    plt.figure(figsize=(6,4))
    bars = plt.bar(names, vals, color=['purple', 'green'])
    plt.ylim(0, 1)
    plt.title("Accuracy Comparison")
    plt.ylabel("Accuracy")

    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v*100:.2f}%", ha='center')

    st.pyplot(plt.gcf())
    plt.close()


# -------------------------------
# Sidebar – Upload + Train
# -------------------------------
st.sidebar.header("⚙️ Setup & Train")
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:
    raw_data = uploaded_file.read()
    detected = chardet.detect(raw_data)
    encoding = detected["encoding"] or "utf-8"
    st.sidebar.info(f"Detected Encoding: {encoding}")

    df = pd.read_csv(pd.io.common.BytesIO(raw_data), encoding=encoding)
    st.write("### Preview of Uploaded Dataset")
    st.dataframe(df.head())

    target_col = st.sidebar.selectbox("Select Target Column", df.columns)
    text_col = st.sidebar.selectbox("Select Text Column", [c for c in df.columns if c != target_col])
    test_size = st.sidebar.slider("Test Size Split", 0.1, 0.5, 0.2)

    if st.sidebar.button("🚀 Train Models"):
        X = df[text_col].astype(str)
        le = LabelEncoder()
        y = le.fit_transform(df[target_col])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Custom model uses TF-IDF
        vectorizer_custom = TfidfVectorizer(max_features=2000, stop_words="english")
        X_train_vec_custom = vectorizer_custom.fit_transform(X_train)
        X_test_vec_custom = vectorizer_custom.transform(X_test)

        model_custom = HardcodedMultinomialNB()
        model_custom.fit(X_train_vec_custom, y_train)
        y_pred_custom = model_custom.predict(X_test_vec_custom)
        acc_custom = accuracy_score(y_test, y_pred_custom)

        # Sklearn model uses CountVectorizer (1,2)
        vectorizer_sk = CountVectorizer(max_features=2000, stop_words="english", ngram_range=(1, 2))
        X_train_vec_sk = vectorizer_sk.fit_transform(X_train)
        X_test_vec_sk = vectorizer_sk.transform(X_test)

        model_sk = MultinomialNB()
        model_sk.fit(X_train_vec_sk, y_train)
        y_pred_sk = model_sk.predict(X_test_vec_sk)
        acc_sk = accuracy_score(y_test, y_pred_sk)

        # Store
        st.session_state.trained = True
        st.session_state.class_names = le.classes_.tolist()
        st.session_state.X_train_text = X_train
        st.session_state.X_test_text = X_test.reset_index(drop=True)
        st.session_state.y_test = y_test

        st.session_state.model_custom = model_custom
        st.session_state.vectorizer_custom = vectorizer_custom
        st.session_state.y_pred_custom = y_pred_custom
        st.session_state.accuracy_custom = acc_custom

        st.session_state.model_sk = model_sk
        st.session_state.vectorizer_sk = vectorizer_sk
        st.session_state.y_pred_sk = y_pred_sk
        st.session_state.accuracy_sk = acc_sk

        st.sidebar.success("✅ Both Models Trained!")

# -------------------------------
# Sidebar Navigation
# -------------------------------
page = st.sidebar.radio("📄 View", ["Custom Model", "Sklearn Model", "Compare"])

if page == "Custom Model":
    render_model_page("custom", "🧪 Custom Hardcoded Multinomial NB (TF-IDF)")
elif page == "Sklearn Model":
    render_model_page("sk", "⚙️ Sklearn Multinomial NB (Count 1,2)")
else:
    render_compare_page()
