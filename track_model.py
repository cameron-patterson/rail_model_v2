parameters = {"z_ins": 0.0289,
              "z_cont": 0.0289,
              "y_ins": 0.1,
              "y_cont": 1.6,
              "v_power": 10,
              "r_power": 7.2,
              "r_relay": 20,
              "r_cb": 1e-3,
              "r_shunt": 251e-4}

# convert to current sources and admittances
i_power = parameters["v_power"] / parameters["r_power"]  # Track circuit power supply current
y_power = 1 / parameters["r_power"]  # Track circuit power supply admittance
y_relay = 1 / parameters["r_relay"]  # Track circuit relay admittance
y_cb = 1 / parameters["r_cb"]  # Cross bond admittance
y_shunt = 1 / parameters["r_shunt"]  # Train shunt admittance

