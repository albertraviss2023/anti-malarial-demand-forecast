# CI/CD Branch 

This main branch hosts the pipeline for **automated forecasting of anti-malarial demand** and **district-level malaria cases**. The system is fully automated through **GitHub Actions**, which:  
- Run monthly prediction jobs  
- Update the trained models’ outputs  
- Merge results into the respective **dashboard branches** for deployment  

For deployment of the full app stacks, we need to switch to main-production branch.
---

## 🔄 Automation Workflow  

The **main branch** orchestrates the automation.  
GitHub Actions workflows are triggered on schedule to:  
1. Generate updated predictions  
2. Integrate results into dashboards  
3. Deploy automatically without manual intervention  

---

## 📂 Branch Structure  

The workflows rely on two key model directories:  

- **`models/`**  
  Contains trained `.keras` models for **Defined Daily Doses (DDD) demand prediction**  

- **`malaria_models/`**  
  Contains trained `.keras` models for **district-level malaria case predictions**
  - **`scripts/`**  
  Contains **python prediction generating scripts**  
 - **`results/`**  
  Contains **monthly prediction results**
 - **`data/`**  
  Contains **raw data backups used elsewhere for model development**  
---

## 🚀 Deployment  

- Predictions are pushed and merged directly into dashboard branches  
- Dashboards update seamlessly with new results after each workflow run  

---

## ⚙️ Tech Highlights  

- **GitHub Actions** for CI/CD automation  
- **Keras** used for saving deep learning models for forecasting  
- Automated monthly prediction and dashboard refresh

  
