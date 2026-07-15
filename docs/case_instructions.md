Recruitment Practical Case:

**Building end-to-end analytics use case: Building an offer-recommendation model to optimize a CVM campaign using Machine Learning**

**The following skills will be evaluated with the case:**

- Basic knowledge of python programming (or being resourceful enough to complete the guided notebook using AI tools). Note: The use of AI coding assistants (e.g., ChatGPT) is permitted and encouraged. However, during the interview, candidates will be asked to explain specific lines of code from their submission. Failure to do so will result in disqualification, as it will be assumed the work was completed by a third party.
- Capacity to get insights from data, understand lead profiling and corresponding business implications
- Capacity to build business case and cash flow analysis to understand financial implications, offer and product fit, in a Customer Value Management campaign
- Capacity to understand the output of a machine learning model and its metrics, and how it can be integrated into a CVM campaign to recommend the right offer to each customer.
- Capacity to understand features that differentiate propensity to convert and how offer features match user profiles
- Capacity to prepare an executive presentation in PowerPoint to summarize the most relevant findings
- Capacity to communicate the result of those findings by presenting the resulting material

**Critical Warning**

**Asking any AI tool (GPT, Claude, Gemini or others) to run or solve this case autonomously without answering each question step by step is detectable and will result in poor answers and being disqualified from the process. What do we expect? The candidate shall use AI collaboratively to expand your knowledge and capabilities and augment your analytics capabilities.**

**The objective of this case is for the candidate to deeply understand how personalized offers matching customer characteristics result in increased conversion and better value creation. The candidate is also expected to perform a multivariate exploratory analysis to understand relevant characteristics from the customer base and how these characteristics impact conversion to different types of offers.**

**Delivering a presentation fully made by an AI solution full of non-relevant data, nonsensical titles, no real insights will be heavily penalized**

**The case also requires to develop an ML model, that can be built using AI. The result will be measured based on two criteria:**

- **The candidate must fully understand the coded model. Meaning properly understanding the architecture, each parameter selected in the model and the code. It is strongly preferred to have a simpler model fully understood by the candidate that a very complex model that is not well explained**
- **The performance of the model compared to the random and the low cost offer.**

**Instructions**

This case was developed for academic and recruiting purposes. All data has been anonymized or synthetically created. Distribution beyond the specific purpose of the case is forbidden.

CellMov is a converged operator in Latin America. CellMov is the country's market leader in mobile but a distant number two in fixed broadband. Conjoint analysis and consulting projects have identified that pairing mobile service with the right fixed product represents a significant cross-sell opportunity.

Detailed call analysis on selected samples has also shown that conversion is significantly impacted by the first offer presented to a customer. If the initial pitch is not appealing, customers normally hang up the phone.

Carla Henriquez, the new CMO of the company, wants to truly exploit this opportunity. The current process of offer assignment is not very analytical. Offers are mostly assigned depending on monthly promotion strategies, without much personalization. In some months they are offered quasi-randomly.

After many failed experiments and considering that propensity to convert was proven to be elastic (dependent on price), operations managers have decided to focus on offering a single offer: Low Speed Internet and Basic TV.

She has hired you to analyze data and build an AI-based recommender to ensure customers receive the right offer while optimizing for total margin.

**Specific Objectives:**

\- Understand key metrics in a customer base management campaign: Price per converted user, margin per converted user, total revenue, total margin, conversion rates.

\- Understand how customer profiling impacts conversion propensity. Model how different user profiles can be paired with specific offer features to maximize conversion and total capture value

\- Understand how to select a target function that maximizes value in a customer base management campaign

\- Develop a machine learning recommender model to optimize the offer presented to each customer.

\- Effectively use AI to develop this case

\- Propose the implementation of an end-to-end AI use case based on the findings from this case and correctly assess why it adds incremental value.

**Inputs:**

The operation has provided two inputs:

1\. Notebook "CVM Simulation - Case.ipynb" to be completed by candidate.

2\. Supporting functions under the folder data_and_supporting_code (base, experiments, sim)

**Output:**

The output of the analysis shall be:

1\. A PowerPoint presentation including:

**a. ML Model implementation results:**

Results from the offer personalization model:

\- Profiling of customer leads by relevant features

\- Assessment of the results from the two experiments (random offer and low cost offer). Description of relevant features that impact propensity to convert. Inference of the match between customer characteristics and offer features that resonate based on the experiment results. Inference of the price impact in the conversion propensity, understanding of the elasticity of the demand for different types of users. Clearly present insights (not data) on how matching offer and customer characteristics drive conversion. This section should be solely devoted to explaining the problem from a business perspective, we will evaluate your understanding of the business problem. **Adding data to the presentation that does not represent any relevant insight, having nonsensical titles, or irrelevant charts (especially if produced by AI) will result in heavy penalization.**

\- Comprehensive explanation of your model architecture, solution strategy. **Make sure to present one slide explaining how your model works and justifying your parameter selection.** **Critical warning:** This case has been tested with several AI solutions. AI support is permitted (and encouraged), but the candidate **must fully understand, explain, and be able to modify any code provided.** During the presentation, the candidate may be required to modify the code live. Failing to explain the model architecture or the model itself will result in disqualification from the process. A solution that is fully understood and can be confidently modified is strongly preferred over a highly complex solution the candidate cannot explain.

\- Comparison of the 3 results: Random offer, low cost offer and optimized results.

**b. How you used AI to develop the case**

An explanation on how AI was used to develop this case, what tools were used, and how AI was used to improve your productivity while answering the case

**c. Agentic AI Use Case Proposal**

Based on your findings from the CVM analytics case, propose an end-to-end AI based case, including Agentic AI to improve the CVM campaign operation.

Your proposal should demonstrate how a system of AI agents can enhance decision-making and optimize financial outcomes.

Detail at least: i) Specific processes to be transformed ii) A high level system architecture of the solution detailing technological tools (integrations, systems, etc) required for the use case iii) Details on how agents would be used. Define how you will measure the incremental value from the Agentic implementation.

2\. If an AI tool was used to develop the model, the prompt log that was used to reach the solution

3\. The completed notebook delivered in BOTH formats: as an .ipynb file AND exported as .html so the evaluator can review the executed cell outputs without running the notebook

**Assessment:**

Your performance will be assessed based on:

\- Your capacity to identify insights from the key metrics, discriminant and valuable features

\- Your capacity to explain the recommender solution, explain your strategy and the solution design process that you followed to reach your solution

\- Your capacity to understand the code in the solution, even if it was developed using AI tools

\- Your capacity to properly use AI to solve complex problems - relying on it to be more productive - while maintaining full understanding and ownership of the solution

\- Your capacity to design an end-to-end AI solution, using AI to provide real value. Proposing agents that do not add real value or have a real purpose will result in incremental cost for the company, thus, including them in the proposal will be penalized in the evaluation.

\- The case presentation to analytics senior leadership

**Required steps to complete the notebook**

- _You will have one week to complete the assignment. Plan your time accordingly - the deadline will not be extended once the notebook is shared._
- Install Anaconda (<https://www.anaconda.com/products/distribution>) or any preferred software to run Python notebooks on your local PC or in a cloud environment. **Candidate may choose alternative environments such as Colab, Sagemaker AI or Visual Studio**
- Save the csv files that have been provided with the case in the same folder that you are running your notebook
- Follow the instructions, get the suggested visualizations, and answer the questions. **Most of the cells are already completed, you should only complete coding or define parameters in cells marked as Coding Cell Read Instructions Carefully**
- **Recommendation: The notebook is made to be completed sequentially, in case you want to repeat or correct a dataframe created by the flow of the notebook, restart kernel and run all cells up to that point.**
- Prepare an executive presentation of maximum 8 content slides to present your answers. Your presentation will be evaluated based on your capacity to present insightful information in a clean, concise and clear way following best industry practices such as the [Pyramid Principle](https://www.theanalystacademy.com/consulting-slide-structure/). We will also evaluate your capacity to properly setup slide layouts and select a color palette that looks professional and helps to deliver the message.
- Present your material to the evaluator in a 30-minute session.