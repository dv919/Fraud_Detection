import pandas as pd
import joblib
import streamlit as st
import sys
import traceback


def load_model(path: str = "fraud_detection_model.pkl"):
	try:
		model = joblib.load(path)
		return model
	except Exception as e:
		msg = f"Failed to load model from {path}: {e}"
		st.error(msg)
		# also print traceback to the terminal for easier debugging
		print(msg, file=sys.stderr)
		traceback.print_exc()
		return None


def main():
	st.title("Fraud Detection Prediction App")

	st.markdown("Please enter the transaction details and use the predict button")

	st.divider()

	transaction_type = st.selectbox(
		"Transaction Type",
		["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"],
	)

	step = st.number_input("Step (time unit)", min_value=1, value=1, step=1)
	amount = st.number_input("Amount", min_value=0.0, value=1000.0)

	oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=0.0)
	newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=0.0)
	oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
	newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)

	model = load_model()

	if st.button("Predict"):
		balanceDiffOrig = oldbalanceOrg - newbalanceOrig
		balanceDiffDest = newbalanceDest - oldbalanceDest

		input_data = pd.DataFrame([
			{
				"step": step,
				"type": transaction_type,
				"amount": amount,
				"oldbalanceOrg": oldbalanceOrg,
				"newbalanceOrig": newbalanceOrig,
				"oldbalanceDest": oldbalanceDest,
				"newbalanceDest": newbalanceDest,
				"balanceDiffOrig": balanceDiffOrig,
				"balanceDiffDest": balanceDiffDest,
			}
		])

		st.write("Input data:")
		st.table(input_data)

		if model is None:
			st.error("Model not available. Place `fraud_detection_pipeline.pkl` in the app folder.")
			return

		try:
			pred = model.predict(input_data)
			if hasattr(model, "predict_proba"):
				proba = model.predict_proba(input_data)
			else:
				proba = None

			if int(pred[0]) == 1:
				st.error("Prediction: Fraudulent transaction")
			else:
				st.success("Prediction: Legitimate transaction")

			if proba is not None:
				# find probability for class 1
				# assume classifier has classes_ with order matching columns of predict_proba
				try:
					classes = list(model.classes_)
					idx = classes.index(1)
					st.write(f"Probability of fraud: {proba[0][idx]:.3f}")
				except Exception:
					# fallback: if binary and proba has two cols, take second
					if proba.shape[1] == 2:
						st.write(f"Probability of fraud: {proba[0][1]:.3f}")
					else:
						st.write("Predicted probabilities:", proba)

		except Exception as e:
			msg = f"Prediction failed: {e}"
			st.error(msg)
			print(msg, file=sys.stderr)
			traceback.print_exc()


if __name__ == "__main__":
	main()