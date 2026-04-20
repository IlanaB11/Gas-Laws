import streamlit as st #for running the website
import pandas as pd
import matplotlib.pyplot as plt

def newton_raphson_approx(x:float, f:function, df:function, depth=0, max_depth=1000, converge_param=1e-5) -> float:
        """Recursive step for newton raphson approximation method (x_n+1 = x_n - f(x_n)/f'(x_n)).
        
        Inputs: 
            x (float): Initial guess for root
            f (function): Function for which we are trying to find a root
            df (function): Derivative of f
            depth (int): Current depth of recursion
            max_depth (int): Maximum depth of recursion before giving up
            converge_param (float): Threshold for convergence 
            
        Output:
            float: Approximation of root of f """
        
        if depth >= max_depth:
            raise RecursionError(f"Did not converge within {max_depth} iterations.")

        residual = f(x) / df(x)

        if abs(residual) < converge_param:  # Converged
            return x 
        
        else: #Not converged
            x_new = x - residual
            return newton_raphson_approx(x_new, f, df, depth + 1, max_depth, converge_param)  # Recursive step
        
def solve_ideal_volume(P: float, T: float = 298, R: float = 0.08206) -> float:
    """
    Calculate the calculate the molar volume of a gas using the ideal gas law 
    
    Inputs:
    P (float): Pressure of the gas (in atm)
    R (float): Ideal gas constant (.08206  L*atm/(mol*K))
    T (float): Temperature of the gas (in Kelvin)
    
    Returns:
        float: Pressure of the gas calculated using the ideal gas law
        """
    
    return (R * T) / P


def solve_vdw_volume(P: float, a: float, b: float, T: float = 298, R: float = 0.08206, converge_param: float = 1e-5, max_depth: int = 1000) -> float:
    """
    Solve the van der Waals equation for volume using the Newton-Raphson method. 

    Inputs:
        P (float): Pressure (atm)
        T (float): Temperature (K)
        a (float): vdW constant a (dm^6·atm/mol^2)
        b (float): vdW constant b (dm^3/mol)
        R (float): Gas constant 
        converge_param (float): Convergence tolerance 
        max_depth (int): max number of iterations 

    Returns:
       float: molar volume V (L) 
    """

    # Cubic coefficients: V³ + c2*V² + c1*V + c0 = 0
    c2 = -(b + (R * T / P))
    c1 = a / P
    c0 = -(a* b) / P

    def f(V):
        """cubic form of van der Waals equation rearranged to f(V) = 0."""
        return V**3 + c2 * V**2 + c1 * V + c0

    def df(V):
        """Derivative of the cubic form of van der Waals equation."""
        return 3 * V**2 + 2 * c2 * V + c1

    # Initial guess - ideal value
    V0 = R * T / P
    return newton_raphson_approx(V0, f, df)

def solve_rk_volume(P: float, A:float, B:float, T:float = 298, R:float = 0.08206, converge_param: float = 1e-5, max_depth: int = 1000)-> float:
    """
    Solve the Redlich-Kwong equation for volume using the Newton-Raphson method. 

    Inputs:
        P (float): Pressure (atm)
        A (float): Redlich-Kwong constant A (dm^6·atm/mol^2)
        B (float): Redlich-Kwong constant B (dm^3/mol)
        T (float): Temperature (K)
        R (float): Gas constant 
        converge_param (float): Convergence tolerance 
        max_depth (int): max number of iterations 

    Returns:
       float: molar volume V (L) 
    """

    # Cubic coefficients: V³ + c2*V² + c1*V + c0 = 0 
    c2 = -(R * T) / P
    c1 = (A / ((T**0.5) * P)) - ((R * T * B) / P) - (B**2)
    c0 = -(A * B) / ((T**0.5) * P)

    def f(V): 
        """cubic form of Redlich-Kwong equation rearranged to f(V) = 0."""
        return V**3 + c2 * V**2 + c1 * V + c0
    
    def df(V): 
        """Derivative of the cubic form of Redlich-Kwong equation."""
        return 3 * V**2 + 2 * c2 * V + c1
    
      # Initial guess - ideal value
    V0 = (R * T / P) + B
    return newton_raphson_approx(V0, f, df)


input_file = st.file_uploader("Upload Data from NIST", type=["txt"], accept_multiple_files = False)

if input_file: 
    data = pd.read_csv(input_file, delimiter="\t",
                                    usecols=[1, 3], 
                                    header=0)
    data.drop(0, inplace=True) #gets rid of the infinity

    pressure_col = data.columns[0]
    data[pressure_col] = data[pressure_col].astype(float)
    data[data.columns[1]] = data[data.columns[1]].astype(float)

    T = st.number_input("Temperature (K)", value = 298.00, step = 0.001, format="%0.3f")
    R = st.number_input("Gas Constant (R)", value = 0.08206, step = 0.00001, format="%0.5f")

    st.header("van der Waals Equation") #van der Waals parameters
    a = st.number_input("a", value = 2.2725, step = 0.00001, format="%0.5f")
    b = st.number_input("b", value = 0.043067, step = 0.00001, format="%0.5f")


    st.subheader("Redlich-Kwong Equation") #rk constants 
    A = st.number_input("b", value = 31.784, step = 0.00001, format="%0.5f")
    B = st.number_input("b", value = 0.029850, step = 0.000001, format="%0.6f")

    for pressure in data[pressure_col]:

        ideal_volume = solve_ideal_volume(pressure, T, R)
        vdw_volume = solve_vdw_volume(pressure, a, b)
        rk_volume = solve_rk_volume(pressure, A, B)

        #write to new column
        data.loc[data[pressure_col] == pressure, "Ideal Volume (l/mol)"] = ideal_volume
        data.loc[data[pressure_col] == pressure, "vdw Volume (l/mol)"] = vdw_volume
        data.loc[data[pressure_col] == pressure, "Rk Volume (l/mol)"] = rk_volume

    gas_fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(data[pressure_col], data[data.columns[1]], label="Experimental")
    ax.scatter(data[pressure_col], data[data.columns[2]], label="Ideal")
    ax.scatter(data[pressure_col], data[data.columns[3]], label="vdW")
    ax.scatter(data[pressure_col], data[data.columns[4]], label="Rk")
    ax.set_title("Experimental vs van der Waals: Methane at 298K")
    ax.set_xlabel("Pressure (atm)")
    ax.set_ylabel("Volume (l/mol)")
    ax.set_yscale('log')
    ax.legend()

    st.pyplot(gas_fig)
    st.download_button("Download Data as CSV", data.to_csv(index=False).encode(), file_name= "Gas_Data.csv") #download cleaned file 
