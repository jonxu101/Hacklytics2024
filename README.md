# Quantitative Healthcare Investing via "Clinical Trial Backed Securities"
---
#### TL;DR This is a project for Hacklytics 2024, where we develop a model for assessing the success rate of cancer-related clinical trials. In our interactive web-app (hosted [here](https://ctbs-dash-630c26238800.herokuapp.com/)), we are able to provide an optimal portfolio construction that maximizes expected return for a user-input risk tolerance.
---
## Background
Quants. The Rocket Scientists of Wall Street. The math geeks and PhDs wield cutting-edge technology to predict financial markets' movements. Indeed, Jonathan and I are on the path to becoming quants, having already dipped our toes in the industry waters—Jonathan as a quantitative developer at a prominent high-frequency trading firm, and myself as a quantitative researcher devising systematic investment strategies. However, a recurring question challenges our professional aspirations: beyond "generating deep liquidity" and "creating more efficient markets," do quants contribute anything of real value to society?

This curiosity propelled us into Hacklytics, not merely to compete but to explore how our quantitative skills could extend beyond financial markets to foster societal good. We asked ourselves, "Can we apply our expertise in data science and quantitative finance to not only predict markets but also promote social impact?" This project is our journey to discover that intersection.

Throughout the end of the 1900s and into the 21st century, modern medicine has rapidly progressed with constant innovation in the research of new drugs. However, it has become increasingly difficult to develop novel effective drugs, driving costs of research & development in healthcare up as time passes. In fact, the number of drugs produced per dollar spent has been decreasing exponentially, sometime deemed Eroom's Law: the inverse of Moore's Law! 

Today, only around 5% of clinical trials reach Phase 4-approval for public use, making investing in them like betting $200 million on a home run--but you can only bet on one bat! As a result, funding for clinical trials has become a risky prospect for any potential investor, but what if we can utilize the power of quantitative investing to make these investments risk-averse?

---

## Dataset & Modeling
In this project, we aggregate cancer clinical trial data from [clinicaltrials.gov](http://clinicaltrials.gov/). We queried for keyword 'cancer', which resulted in a dataset with approximately 60,000 rows. In addition to quantitative parameters such as sponsors (via one-hot encoding) and number of participants, we incorporated qualitative data via machine learning. We utilized Hugging Face's sentence_transformer API in order to extract sentence vector representations, giving additional parameters to train on.

We apply a Multilayer Perception (MLP) model on the feature-engineered dataset 

---

## Portfolio Optimization
With our predicted success rate and returns, we now apply Markowitz Mean-Variance Optimization to solve for optimal weightings of each clinical trial. For a given risk tolerance $\sigma$, we may solve the following system:
$$\underset{w}{\max} w^T\hat{\mu}$$
$$\text{s.t.}$$
$$w^T\hat{\Sigma}w \le \sigma$$
$$\sum w_i = 1$$
This a linear programming problem, and we utilize scipy and pypfopt to optimize these constraints. By generating the maximal expected return over a series of risk tolerances, we are able to produce the Efficient Frontier, 

---

## Interactive Dashboard
Our interactive dashboard found [here](https://ctbs-dash-630c26238800.herokuapp.com/) displays the recommended porfolio distribution given 1000 clinical trials as a test set. The interactive slider allows the user to specify a risk tolerance. After specifying a risk tolerance, the website automatically updates the elements to reflect the optimal portfolio construction.

The data table and pie chart show the distribution that maximizes returns for the given risk tolerance. The data table features basic information the clinical trials sorted by the highest weighted elements, with a clickable hotlink to the full information found on clinicaltrials.gov.

At the bottom of the page is a graph that illustrates the Efficient Frontier as the expected returns over the allowed risk. The EFficient Frontier is plotted as an orange line, along with the expected returns of several randomly generated suboptimal weightings. The red point represents the user selected risk tolerance, which updates accordingly on the graph everytime the user changes the risk tolerance.