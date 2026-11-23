import openmdao.api as om
import math
import numpy as np
import scipy


class Battery(om.ExplicitComponent):

    def setup(self):

        self.add_input('t_eclipse', val=60, units="min")
        self.add_input('P_wheel', units="W")
        self.add_input('DoD', val=0.1)
        self.add_input('e_battery', val=50, units="W*h/kg") # NiCd battery

        self.add_output('E_battery')
        self.add_output('m_battery', units="kg")

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):

        t_eclipse = inputs['t_eclipse']
        P_wheel = inputs['P_wheel']
        DoD = inputs['DoD']
        e_battery = inputs['e_battery']

        Margin = 1
        E_eclipse = (1+Margin) * P_wheel * (t_eclipse / 60)
        E_battery = E_eclipse / DoD
        m_battery = E_battery / e_battery

        outputs['E_battery'] = E_battery
        outputs['m_battery'] = m_battery


class MomentumWheel(om.ExplicitComponent):

    def setup(self):
        self.add_input('T_orbit', val=96, units="min", desc="Orbit period")
        self.add_input('tau_required', val=0.00001, units="N*m", desc="Total required sizing torque")
        self.add_input('r_wheel', val=0.02, units="m", desc="Radius disc area")
        self.add_input('omega_wheel', val=523.6, units="rad/s", desc="Angular velocity")
        self.add_input('theta_allowable', val=0.07, units="rad", desc="Allowable angle displacement")
        # self.add_input('I_satellite', units="kg*m**2", desc="Inertial moment of satellite")

        self.add_output('L_required', units="kg*m**2/s", desc="Angular momentum required")
        self.add_output('m_wheel', units="kg", desc="Momentum wheel mass")
        self.add_output('P_wheel', units="W", desc="Power required by the momentum wheel")

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        T_orbit = inputs['T_orbit']
        tau_required = inputs['tau_required']
        r_wheel = inputs['r_wheel']
        omega_wheel = inputs['omega_wheel']
        theta_allowable = inputs['theta_allowable']
        # I_satellite = inputs['I_satellite']

        # tau_required = I_satellite * 0.00035

        outputs['L_required'] = tau_required * (T_orbit * 60 / 4) * (theta_allowable * 180 / math.pi)
        outputs['m_wheel'] = 2 * outputs['L_required'] /  (r_wheel ** 2) / (omega_wheel * 2 * math.pi / 60)
        outputs['P_wheel'] = outputs['L_required'] * outputs['m_wheel']


class Structure(om.ExplicitComponent):

    def setup(self):
        self.add_input('m_battery', units="kg", desc="mass of battery")
        self.add_input('m_wheel', units="kg", desc="mass of momentum wheel")

        self.add_output('m_satellite', units="kg", desc="mass of satellite")
        self.add_output('cost_satellite', desc="cost of satellite")
        # self.add_output('I_satellite', units="kg*m**2", desc="inertial moment of satellite")

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        m_battery = inputs['m_battery']
        m_wheel = inputs['m_wheel']

        m_satellite = m_battery + m_wheel
        # I_satellite = (1/6) * m_satellite * 0.083**2

        cost_satellite = m_battery*50 + m_wheel*2

        outputs['m_satellite'] = m_satellite
        # outputs['I_satellite'] = I_satellite
        outputs['cost_satellite'] = cost_satellite



if __name__ == "__main__":

    prob = om.Problem()
    satellite = prob.model
    satellite.add_subsystem('MomentumWheel', MomentumWheel(), promotes_inputs=['*'], promotes_outputs=['*'])
    satellite.add_subsystem('Battery', Battery(), promotes_inputs=['*'], promotes_outputs=['*'])
    satellite.add_subsystem('Structure', Structure(), promotes_inputs=['*'], promotes_outputs=['*'])

    # For MDO
    satellite.add_design_var('r_wheel', lower=0.01, upper=0.05)
    satellite.add_design_var('theta_allowable', lower=0.04, upper=0.1)
    # satellite.add_design_var('e_battery', lower=1, upper=10)
    satellite.add_constraint('cost_satellite', upper=2)
    satellite.add_objective('m_satellite', scaler=1)
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'

    prob.setup()
    prob.run_driver()

    prob.model.list_inputs(val=True, units=True)
    prob.model.list_outputs(val=True, units=True)
