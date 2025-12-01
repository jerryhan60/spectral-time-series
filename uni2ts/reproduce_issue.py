
import numpy as np
from uni2ts.transform.precondition import PolynomialPrecondition, ReversePrecondition

def test_forecast_reversal_issue():
    print("Testing ReversePrecondition on forecast window...")
    
    # Setup
    degree = 2
    # c1 = 0, c2 = -0.5 for Chebyshev degree 2
    # y_t = y~_t - (0*y_{t-1} - 0.5*y_{t-2}) = y~_t + 0.5*y_{t-2}
    
    # Create a full sequence
    # t: 0, 1, 2, 3, 4
    # y: 1, 1, 1, 1, 1
    full_y = np.ones(5)
    
    # Apply preconditioning
    precond = PolynomialPrecondition(polynomial_type="chebyshev", degree=degree)
    data = {"target": full_y.copy()}
    data = precond(data)
    full_y_tilde = data["target"]
    coeffs = data["precondition_coeffs"]
    
    print(f"Original y: {full_y}")
    print(f"Preconditioned y~: {full_y_tilde}")
    # Expected:
    # t=0,1 (<=n): y~ = y = 1
    # t=2: y~_2 = y_2 + (-0.5)*y_0 = 1 - 0.5 = 0.5
    # t=3: y~_3 = y_3 + (-0.5)*y_1 = 1 - 0.5 = 0.5
    # t=4: y~_4 = y_4 + (-0.5)*y_2 = 1 - 0.5 = 0.5
    
    # Simulate Forecasting
    # Assume we observed t=0,1,2 (input) and want to forecast t=3,4
    # The model perfectly predicts y~_3, y~_4
    input_window = full_y[:3] # y_0, y_1, y_2
    forecast_y_tilde = full_y_tilde[3:] # y~_3, y~_4
    
    print(f"\nInput window (y): {input_window}")
    print(f"Forecast (y~): {forecast_y_tilde}")
    
    # Try to reverse using ReversePrecondition on the forecast ONLY
    print("\nAttempting reversal on forecast only WITH CONTEXT...")
    data_pred = {
        "prediction": forecast_y_tilde.copy(),
        "precondition_coeffs": coeffs,
        "precondition_degree": degree,
        "precondition_enabled": True,
        "precondition_type": "chebyshev"
    }
    
    reverse = ReversePrecondition(prediction_field="prediction")
    # Pass context (input_window) to the transform
    data_pred = reverse(data_pred, context=input_window)
    reversed_forecast = data_pred["prediction"]
    
    print(f"Reversed forecast: {reversed_forecast}")
    
    # Expected correct reversal:
    # y_3 = y~_3 - (-0.5)*y_1 = 0.5 + 0.5*1 = 1.0
    # y_4 = y~_4 - (-0.5)*y_2 = 0.5 + 0.5*1 = 1.0
    
    print(f"Correct forecast: {full_y[3:]}")
    
    if not np.allclose(reversed_forecast, full_y[3:]):
        print("\nFAILURE: Reversed forecast does not match ground truth!")
    else:
        print("\nSUCCESS: Reversed forecast matches ground truth!")

if __name__ == "__main__":
    test_forecast_reversal_issue()
