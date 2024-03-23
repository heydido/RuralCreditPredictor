![RuralCredit](RuralCredit.jpg)

Photo by <a href="https://unsplash.com/@tomcchen?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Tom Chen</a> on <a href="https://unsplash.com/photos/woman-leaning-on-wall-jO1OyKR7s68?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Unsplash</a>

# What is RuralCredit?
RuralCredit fosters economic empowerment in rural India by providing accessible financial services tailored to local needs. Through microfinance, agricultural loans, and financial literacy programs, it cultivates entrepreneurship, boosts agricultural productivity, and enhances livelihoods. RuralCredit aims to bridge the financial gap, unlocking the potential for sustainable development in rural communities.

------------------------------------------------------------------------------------------------------------------------
## Setting Up the Project:
1. Clone the repository
2. Create a virtual environment using setup.sh:
    ```
    bash setup.sh 
    ```
3. Activate the virtual environment (optional, if not done in step 2)
    ```
    source activate ./venv
    ``` 
   
## Workflow:
1. Update config: `config/config.yaml`
2. Update raw/processed data schema: `raw_schema.yaml/processed_schema.yaml` (if needed)
3. Update model parameters: `params.yaml` (if needed)
4. Update the entity: `src/RuralCreditPredictor/entity/config_entity.py`
5. Update the configuration manager: `src/RuralCreditPredictor/config/configuration.py`
6. Update the components: `src/RuralCreditPredictor/components`
7. Update the pipeline: `src/RuralCreditPredictor/pipeline`
8. Update entrypoint: `main.py`
9. Update application: `app.py`