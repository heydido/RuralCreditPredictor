# RuralCreditPredictor
RuralCreditPredictor: Empowering Tomorrow's Opportunities

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
1. Update config: config/config.yaml
2. Update raw data schema: schema.yaml (if needed)
3. Update model parameters: params.yaml (if needed)
4. Update the entity: src/RuralCreditPredictor/entity/config_entity.py
5. Update the configuration manager: src/RuralCreditPredictor/config/configuration.py
6. Update the components: src/RuralCreditPredictor/components
7. Update the pipeline: src/RuralCreditPredictor/pipeline
8. Update entrypoint: main.py
9. Update application: app.py