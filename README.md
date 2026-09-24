# ShopSafe AI

A Streamlit app that classifies URLs as potentially phishing or legitimate using the saved Random Forest model.

The deployment uses the same 17 URL features as the training notebook. It does not open, crawl, or fetch the submitted website.

## Files

```text
my_scam_checker_project/
├── app.py
├── model.pkl
├── features.pkl
├── training.ipynb
├── requirements.txt
└── README.md
```

## Run on macOS

Open Terminal and run:

```bash
cd ~/Downloads/my_scam_checker_project
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Check the model

Try:

```text
google.com
https://www.google.com
amazon.com
https://www.microsoft.com
```

For a bare domain such as `google.com`, the app scores the common `www` form so the URL is represented consistently with the training examples.

## GitHub

```bash
git add .
git commit -m "Clean up URL scam checker"
git push
```

## Notes

The model is a URL-feature baseline. URL-only classification can produce false positives and false negatives, so the result should not be treated as a complete security verdict.
