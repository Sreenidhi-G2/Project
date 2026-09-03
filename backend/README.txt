VeriFace Backend
=================

Setup:
1. Place these files in C:\Data Razorpay\backend\app\
2. Make sure your ml/training folder (dataset.py, model.py) stays at
   C:\Data Razorpay\ml\training - inference.py imports from there.
3. Install dependencies:
   python -m pip install -r requirements.txt
4. Create the output folder:
   mkdir C:\Data Razorpay\backend
5. Run:
   cd C:\Data Razorpay\backend\app
   python -m uvicorn main:app --reload
6. Open http://127.0.0.1:8000/docs to test interactively.
