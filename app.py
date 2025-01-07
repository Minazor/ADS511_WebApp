import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, session
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy.stats import shapiro, levene, kruskal

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.secret_key = "your_secret_key"


# -- Yardımcı Fonksiyonlar -- #
def check_normality(data):
    """Shapiro-Wilk normalite testi. p < 0.05 ise normal değil demek."""
    data = data.dropna()
    if len(data) < 3:
        return None, None  # veri çok azsa test yapma
    w, p = stats.shapiro(data)
    return w, p


def check_variances(data1, data2):
    """Levene testi ile varyans homojenliği."""
    data1 = data1.dropna()
    data2 = data2.dropna()
    stat, p = stats.levene(data1, data2)
    return stat, p


# Dosya yükleme sayfası
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename.endswith(".csv"):
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            # Veriyi session'a koyalım ki sonradan okuyabilelim
            session["filename"] = file.filename
            return redirect(url_for("select_samples"))
        else:
            return "Lütfen geçerli bir CSV dosyası yükleyin."
    return render_template("index.html")


@app.route("/select_samples", methods=["GET", "POST"])
def select_samples():
    """Flow chart'ın ilk sorusu: Kaç örneklem var? (1, 2 veya birden fazla?)"""
    if request.method == "POST":
        sample_choice = request.form.get("sample_choice")  # '1', '2', 'many'
        session["sample_choice"] = sample_choice

        if sample_choice == "1":
            return redirect(url_for("one_sample_decision"))
        elif sample_choice == "2":
            return redirect(url_for("two_sample_decision"))
        else:
            return redirect(url_for("multiple_samples_decision"))
    return render_template("select_samples.html")


# ---------- 1 SAMPLE ----------- #


@app.route("/one_sample_decision", methods=["GET", "POST"])
def one_sample_decision():
    """
    Diyagrama göre:
      1) Known population variance var mı (Z-test) yok mu (T-test)?
      2) Normalite?
         - Normal dağılım yoksa genelde non-parametrik tek örneklem testleri (ör. Wilcoxon Signed Rank "tek örneklem" versiyonu)
           kullanılabilir. Flow chart'ta açık yazmıyor ama biz ekleyebiliriz.
    """
    if request.method == "POST":
        known_variance = request.form.get("known_variance")  # 'yes' or 'no'
        session["known_variance"] = known_variance
        return redirect(url_for("one_sample_columns"))
    return render_template("one_sample_decision.html")


@app.route("/one_sample_columns", methods=["GET", "POST"])
def one_sample_columns():
    """
    Burada kullanıcıdan tek bir numeric kolon (örnek: PM2.5) ve
    bir hipotetik değer (mu0) almasını isteyebiliriz.
    """
    filename = session.get("filename")
    if not filename:
        return redirect(url_for("index"))
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    df = pd.read_csv(filepath)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if request.method == "POST":
        numeric_col = request.form.get("numeric_col")
        hypoth_value = float(request.form.get("hypoth_value", 0))
        session["numeric_col"] = numeric_col
        session["hypoth_value"] = hypoth_value

        # Normaliteyi kontrol edelim
        data = df[numeric_col].dropna()
        w, p_val = check_normality(data)
        session["normality_p"] = p_val

        return redirect(url_for("one_sample_test"))

    return render_template("one_sample_columns.html", numeric_cols=numeric_cols)


@app.route("/one_sample_test")
def one_sample_test():
    filename = session.get("filename")
    numeric_col = session.get("numeric_col")
    hypoth_value = float(session.get("hypoth_value", 0))
    known_variance = session.get("known_variance")
    normality_p = float(
        session.get("normality_p", 1.0)
    )  # Varsayılan 1.0 (normal kabul edilir)

    if not filename or numeric_col is None:
        return redirect(url_for("index"))

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    df = pd.read_csv(filepath)
    data = df[numeric_col].dropna()
    alpha = 0.05

    result_text = f"One-Sample Test for {numeric_col} vs μ₀={hypoth_value}\n"

    if known_variance == "yes":
        # Known Population Variance varsa, Z-test uygulaması
        sigma = 1.0  # Varsayılan bir sigma değeri, gerçekte dışarıdan gelmelidir
        z_stat = (np.mean(data) - hypoth_value) / (sigma / np.sqrt(len(data)))
        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        result_text += f"One-Sample Z-test: Z={z_stat:.4f}, p={p_val:.4f}\n"
    else:
        # Known Variance yoksa, normalite kontrolü ile devam
        if normality_p < alpha:
            # Normal dağılım değil => non-parametrik test
            diff = data - hypoth_value
            stat, pval = stats.wilcoxon(diff)
            result_text += f"Non-parametrik Wilcoxon Signed-Rank uygulanıyor.\n"
            result_text += f"stat={stat:.4f}, p={pval:.4f}\n"
        else:
            # Normal dağılım => T-test
            stat, pval = stats.ttest_1samp(data, hypoth_value)
            result_text += f"One-Sample T-test: T={stat:.4f}, p={pval:.4f}\n"

    return render_template("result.html", result=result_text)


# ---------- 2 SAMPLES ----------- #


@app.route("/two_sample_decision", methods=["GET", "POST"])
def two_sample_decision():
    """
    1) Paired samples? => Evet/Hayır
    2) Normal dağılım? => (Assumption check)
    """
    if request.method == "POST":
        is_paired = request.form.get("is_paired")  # 'yes' / 'no'
        session["is_paired"] = is_paired
        return redirect(url_for("two_sample_columns"))
    return render_template("two_sample_decision.html")


@app.route("/two_sample_columns", methods=["GET", "POST"])
def two_sample_columns():
    """
    - numeric kolon(lar) seçimi
    - eğer paired ise, muhtemelen 'before' ve 'after' veya benzeri 2 numeric kolon
    - eğer independent ise, 1 numeric kolon ve 1 kategori (2 grup) seçilir.
    """
    filename = session.get("filename")
    if not filename:
        return redirect(url_for("index"))
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    df = pd.read_csv(filepath)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    is_paired = session.get("is_paired")

    if request.method == "POST":
        if is_paired == "yes":
            # Kullanıcı 2 numeric kolon seçsin
            numeric_col_1 = request.form.get("numeric_col_1")
            numeric_col_2 = request.form.get("numeric_col_2")
            session["numeric_col_1"] = numeric_col_1
            session["numeric_col_2"] = numeric_col_2
        else:
            # Kullanıcı 1 numeric ve 1 categorical (2 grup)
            numeric_col = request.form.get("numeric_col")
            cat_col = request.form.get("cat_col")
            session["numeric_col"] = numeric_col
            session["cat_col"] = cat_col

        return redirect(url_for("two_sample_test"))

    return render_template(
        "two_sample_columns.html",
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        is_paired=is_paired,
    )


@app.route("/two_sample_test")
def two_sample_test():
    """
    2 sample test uygulama:
      - Paired => normal => paired t-test, normal değil => Wilcoxon Signed Rank
      - Independent => normal & var.homogeneous => two-sample t-test, yoksa mann-whitney
    """
    filename = session.get("filename")
    is_paired = session.get("is_paired")

    if not filename or is_paired not in ["yes", "no"]:
        return redirect(url_for("index"))

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    df = pd.read_csv(filepath)

    alpha = 0.05
    result_text = ""

    if is_paired == "yes":
        col1 = session.get("numeric_col_1")
        col2 = session.get("numeric_col_2")
        data1 = df[col1].dropna()
        data2 = df[col2].dropna()

        # Normalite?
        w1, p1 = check_normality(data1)
        w2, p2 = check_normality(data2)
        normal = True
        if (p1 is not None and p1 < alpha) or (p2 is not None and p2 < alpha):
            normal = False

        if normal:
            # Paired T-test
            stat, pval = stats.ttest_rel(data1, data2)
            result_text += (
                f"Paired T-test ({col1} vs {col2}): t={stat:.4f}, p={pval:.4f}"
            )
        else:
            # Wilcoxon Signed Rank
            stat, pval = stats.wilcoxon(data1, data2)
            result_text += f"Wilcoxon Signed-Rank Test (Paired, non-param): stat={stat:.4f}, p={pval:.4f}"

    else:
        # Independent
        numeric_col = session.get("numeric_col")
        cat_col = session.get("cat_col")
        unique_vals = df[cat_col].dropna().unique()
        if len(unique_vals) != 2:
            return "Bu kategorik kolon 2 farklı değer içermiyor!"

        g1, g2 = unique_vals[0], unique_vals[1]
        data1 = df[df[cat_col] == g1][numeric_col].dropna()
        data2 = df[df[cat_col] == g2][numeric_col].dropna()

        # Normalite
        w1, p1 = check_normality(data1)
        w2, p2 = check_normality(data2)
        normal = True
        if (p1 is not None and p1 < alpha) or (p2 is not None and p2 < alpha):
            normal = False

        # Varyans homojenliği
        stat_levene, p_levene = check_variances(data1, data2)
        equal_var = True
        if p_levene < alpha:
            equal_var = False

        if normal:
            # Two-sample T-test
            stat, pval = stats.ttest_ind(data1, data2, equal_var=equal_var)
            result_text += f"Two-sample T-test (independent) {g1} vs {g2}, T={stat:.4f}, p={pval:.4f}, equal_var={equal_var}"
        else:
            # Mann-Whitney
            stat, pval = stats.mannwhitneyu(data1, data2, alternative="two-sided")
            result_text += (
                f"Mann-Whitney U Test: {g1} vs {g2}, U={stat:.4f}, p={pval:.4f}"
            )

    return render_template("result.html", result=result_text)


# ---------- >2 SAMPLES (ANOVA vb.) ----------- #


@app.route("/multiple_samples_decision", methods=["GET", "POST"])
def multiple_samples_decision():
    """
    Diyagramda '2 veya daha fazla sample' -> 1 variable / 2 variables
    => 1 variable => One-way ANOVA vs. Kruskal-Wallis
    => 2 variables => Two-way ANOVA vb.
    """
    if request.method == "POST":
        design_choice = request.form.get("design_choice")  # 'one_var' / 'two_var'
        session["design_choice"] = design_choice
        return redirect(url_for("multi_sample_columns"))
    return render_template("multiple_samples_decision.html")


@app.route("/multi_sample_columns", methods=["GET", "POST"])
def multi_sample_columns():
    """
    - Tek yönlü ANOVA için: 1 numeric, 1 categorical (3+ grup)
    - Normalite/varyans homojenliği? => ANOVA veya Kruskal-Wallis
    - İki yönlü ANOVA (two-way) vs. Friedman vs. Randomized Block vb.
    """
    filename = session.get("filename")
    if not filename:
        return redirect(url_for("index"))
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    df = pd.read_csv(filepath)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    design_choice = session.get("design_choice", "one_var")

    if request.method == "POST":
        if design_choice == "one_var":
            numeric_col = request.form.get("numeric_col")
            cat_col = request.form.get("cat_col")
            session["numeric_col"] = numeric_col
            session["cat_col"] = cat_col
        else:
            # two_var senaryosu -> 2 kategorik kolon + 1 numeric
            numeric_col = request.form.get("numeric_col")
            factor1 = request.form.get("factor1")
            factor2 = request.form.get("factor2")
            session["numeric_col"] = numeric_col
            session["factor1"] = factor1
            session["factor2"] = factor2

        return redirect(url_for("multi_sample_test"))

    return render_template(
        "multi_sample_columns.html",
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        design_choice=design_choice,
    )


def run_two_way_anova(df, numeric_col, factor1, factor2):
    # Two-way ANOVA için model oluştur
    formula = f"{numeric_col} ~ C({factor1}) + C({factor2}) + C({factor1}):C({factor2})"
    model = ols(formula, data=df).fit()
    anova_results = anova_lm(model)
    return anova_results


def run_one_way_anova(df, numeric_col, cat_col):
    # One-way ANOVA için model oluştur
    formula = f"{numeric_col} ~ C({cat_col})"
    model = ols(formula, data=df).fit()
    anova_results = anova_lm(model)
    return anova_results


def check_normality(data):
    # Normallik testi (Shapiro-Wilk)
    stat, p_value = shapiro(data)
    return stat, p_value


def check_variance_homogeneity(*groups):
    # Varyans homojenliği testi (Levene)
    stat, p_value = levene(*groups)
    return stat, p_value


@app.route("/multi_sample_test")
def multi_sample_test():
    """
    Tek değişkenli ANOVA veya Kruskal-Wallis,
    İki değişkenli ANOVA vs. Friedman / Randomized Block vb.
    """
    design_choice = session.get("design_choice", "one_var")
    filename = session.get("filename")
    if not filename:
        return redirect(url_for("index"))
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    df = pd.read_csv(filepath)

    alpha = 0.05
    result_text = ""

    if design_choice == "one_var":
        numeric_col = session.get("numeric_col")
        cat_col = session.get("cat_col")
        groups = df[cat_col].dropna().unique()

        # 3+ grup var mı?
        if len(groups) < 3:
            return f"{cat_col} değişkeninde 3'ten az kategori var. Tek yönlü ANOVA için en az 3 grup olmalı."

        # Grupları liste halinde toplayalım
        group_data = [df[df[cat_col] == g][numeric_col].dropna() for g in groups]

        # Normallik kontrolü
        normal_flag = True
        normality_results = []
        for gdata in group_data:
            if len(gdata) < 3:
                continue
            stat, pval = check_normality(gdata)
            normality_results.append((gdata.name, stat, pval))
            if pval is not None and pval < alpha:
                normal_flag = False

        # Varyans homojenliği kontrolü
        homogeneity_flag = True
        levene_stat, levene_pval = None, None
        if len(group_data) >= 2:
            levene_stat, levene_pval = check_variance_homogeneity(*group_data)
            homogeneity_flag = levene_pval > alpha

        # Test seçimi
        if normal_flag and homogeneity_flag:
            # One-way ANOVA
            anova_results = run_one_way_anova(df, numeric_col, cat_col)
            result_text += f"One-Way ANOVA Results:\n{anova_results.to_string()}\n\n"
        else:
            # Kruskal-Wallis
            kruskal_stat, kruskal_pval = kruskal(*group_data)
            result_text += (
                f"Kruskal-Wallis Test: H={kruskal_stat:.4f}, p={kruskal_pval:.4f}\n\n"
            )

        # Ek test sonuçlarını ekle
        result_text += "Normallik Testi (Shapiro-Wilk):\n"
        for group_name, stat, pval in normality_results:
            result_text += f"Grup: {group_name}, W={stat:.4f}, p={pval:.4f}\n"

        result_text += f"\nVaryans Homojenliği Testi (Levene): Stat={levene_stat:.4f}, p={levene_pval:.4f}\n"

    elif design_choice == "two_var":
        # İlgili kolonları al
        numeric_col = session.get("numeric_col")
        factor1 = session.get("factor1")
        factor2 = session.get("factor2")

        if not all([numeric_col, factor1, factor2]):
            return "Gerekli sütunlar seçilmedi. Lütfen tekrar deneyin."

        if (
            numeric_col not in df.columns
            or factor1 not in df.columns
            or factor2 not in df.columns
        ):
            return "Seçilen sütunlar bulunamadı."

        # Grupları oluştur
        groups = df.groupby([factor1, factor2])[numeric_col].apply(list)

        # Eksik verileri temizle ve grupları eşitle
        min_length = min(len(g) for g in groups if len(g) > 0)
        groups = groups.apply(lambda x: x[:min_length] if len(x) > min_length else x)

        # Normallik ve homojenlik kontrolü
        normal_flag = True
        homogeneity_flag = True

        # Normallik kontrolü
        normality_results = []
        for group_data in groups:
            if len(group_data) < 3:
                continue
            stat, pval = check_normality(pd.Series(group_data))
            normality_results.append((group_data, stat, pval))
            if pval is not None and pval < alpha:
                normal_flag = False

        # Varyans homojenliği kontrolü
        levene_stat, levene_pval = None, None
        if len(groups) >= 2:
            levene_stat, levene_pval = check_variance_homogeneity(*groups)
            homogeneity_flag = levene_pval > alpha

        # Test seçimi
        if normal_flag and homogeneity_flag:
            # Randomized Block Design veya ANOVA
            result_text += f"Randomized Block Design test uygulanacak.\n"
            result_text += f"(Not: Özel bir hesaplama eklenebilir.)\n\n"
        else:
            # Friedman's Test
            try:
                friedman_stat, friedman_pval = stats.friedmanchisquare(*groups)
                result_text += f"Friedman's Test: stat={friedman_stat:.4f}, p={friedman_pval:.4f}\n\n"
            except Exception as e:
                result_text += f"Friedman's Test sırasında bir hata oluştu: {e}\n\n"

        # Normallik ve Levene Test Sonuçları
        result_text += "Normallik Testi (Shapiro-Wilk):\n"
        for group_data, stat, pval in normality_results:
            result_text += f"Grup: {group_data[:10]}..., W={stat:.4f}, p={pval:.4f}\n"

        result_text += f"\nVaryans Homojenliği Testi (Levene): Stat={levene_stat:.4f}, p={levene_pval:.4f}\n"

    else:
        result_text = "Geçersiz seçim. Lütfen tekrar deneyin."

    return render_template("result.html", result=result_text)


# ----------------------

if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)
