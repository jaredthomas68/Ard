import warnings
import numpy as np

import openmdao.api as om
from wisdem.plant_financese.plant_finance import PlantFinance as PlantFinance_orig
from wisdem.landbosse.landbosse_omdao.landbosse import LandBOSSE as LandBOSSE_orig
from wisdem.orbit.orbit_api import Orbit as Orbit_orig

from ard.cost.approximate_turbine_spacing import SpacingApproximations

class LandBOSSEWithSpacingApproximations(om.Group):
    """
    OpenMDAO group that connects the SpacingApproximations component to the LandBOSSE component.

    This group calculates the turbine spacing using the SpacingApproximations and passes it
    to the LandBOSSE component for further cost estimation.
    """

    def initialize(self):
        """Initialize the group and declare options."""
        self.options.declare(
            "modeling_options", types=dict, desc="Ard modeling options"
        )

    def setup(self):
        """Set up the group by adding and connecting components."""
        # Add the PrimarySpacingApproximations component
        self.add_subsystem(
            "spacing_approximations",
            SpacingApproximations(modeling_options=self.options["modeling_options"]),
            promotes_inputs=["total_length_cables"],
        )

        # Add the LandBOSSE component
        self.add_subsystem(
            "landbosse",
            LandBOSSEGroup(modeling_options=self.options["modeling_options"]),
            promotes_inputs=[
                "*",
                (
                    "turbine_spacing_rotor_diameters",
                    "internal_turbine_spacing_rotor_diameters",
                ),
                (
                    "row_spacing_rotor_diameters",
                    "internal_row_spacing_rotor_diameters",
                ),
            ],
            promotes_outputs=["*"],  # Expose all outputs from LandBOSSE
        )

        # Connect the turbine and row spacing outputs from the approximations to LandBOSSE
        self.connect(
            "spacing_approximations.primary_turbine_spacing_diameters",
            "internal_turbine_spacing_rotor_diameters",
        )

        self.connect(
            "spacing_approximations.secondary_turbine_spacing_diameters",
            "internal_row_spacing_rotor_diameters",
        )

class LandBOSSEGroup(om.Group):

    def initialize(self):
        """Initialize the group and declare options."""
        self.options.declare(
            "modeling_options", types=dict, desc="Ard modeling options"
        )
    
    def setup(self):

        # add IVCs for landbosse
        variable_mapping = LandBOSSE_setup_latents(modeling_options=self.options["modeling_options"])

        # create source independent variable components for LandBOSSE inputs
        for key, val in variable_mapping.items():
            if key in ["num_turbines", "number_of_blades"]:
                comp = om.IndepVarComp()
                comp.add_discrete_output(name=key, val=val)
                self.add_subsystem(f"IVC_landbosse_{key}", comp, promotes=["*"])
            else:
                self.add_subsystem(f"IVC_landbosse_{key}", om.IndepVarComp(key, val=val), promotes=["*"])

        # add landbosse
        self.add_subsystem("landbosse", LandBOSSE_orig(), promotes=[
                "total_capex_kW",
                "turbine_spacing_rotor_diameters",
                "row_spacing_rotor_diameters",
            ],
        )

        # connect 
        for key, val in variable_mapping.items():
            self.connect(key, f"landbosse.{key}")

class ORBITGroup(om.Group):

    def initialize(self):
        """Initialize the group and declare options."""
        self.options.declare(
            "modeling_options", types=dict, desc="Ard modeling options"
        )
    
    def setup(self):

        # add IVCs for landbosse
        variable_mapping = ORBIT_setup_latents(modeling_options=self.options["modeling_options"])

        # create source independent variable components for LandBOSSE inputs
        for key, val in variable_mapping.items():
            if key in ["num_turbines", "number_of_blades"]:
                comp = om.IndepVarComp()
                comp.add_discrete_output(name=key, val=val)
                self.add_subsystem(f"IVC_orbit_{key}", comp, promotes=["*"])
            else:
                self.add_subsystem(f"IVC_orbit_{key}", om.IndepVarComp(key, val=val), promotes=["*"])

        # add orbit #TODO check promotes
        self.add_subsystem("orbit", Orbit_orig(), promotes=[
                "total_capex_kW",
                "turbine_spacing_rotor_diameters",
                "row_spacing_rotor_diameters",
            ],
        )

        # connect 
        for key, val in variable_mapping.items():
            self.connect(key, f"orbit.{key}")

class FinanceSEGroup(om.Group):

    def initialize(self):
        """Initialize the group and declare options."""
        self.options.declare(
            "modeling_options", types=dict, desc="Ard modeling options"
        )
    
    def setup(self):

        # add IVCs for landbosse
        variable_mapping = FinanceSE_setup_latents(modeling_options=self.options["modeling_options"])

        # create source independent variable components for LandBOSSE inputs
        for key, val in variable_mapping.items():
            if key in ["num_turbines", "number_of_blades"]:
                comp = om.IndepVarComp()
                comp.add_discrete_output(name=key, val=val)
                self.add_subsystem(f"IVC_financese_{key}", comp, promotes=["*"])
            else:
                self.add_subsystem(f"IVC_financese_{key}", om.IndepVarComp(key, val=val), promotes=["*"])

        # add orbit #TODO check promotes
        self.add_subsystem("financese", Orbit_orig(), promotes=[
                "total_capex_kW",
                "turbine_spacing_rotor_diameters",
                "row_spacing_rotor_diameters",
            ],
        )

        # connect 
        for key, val in variable_mapping.items():
            self.connect(key, f"financese.{key}")

class TurbineCapitalCosts(om.ExplicitComponent):
    """
    A simple component to compute the turbine capital costs.

    Inputs
    ------
    machine_rating : float
        rating of the wind turbine in kW
    tcc_per_kW : float
        turbine capital costs per kW (as output from WISDEM tools)
    offset_tcc_per_kW : float
        additional tcc per kW (offset)

    Discrete Inputs
    ---------------
    turbine_number : int
        number of turbines in the farm

    Outputs
    -------
    tcc : float
        turbine capital costs in USD
    """

    def setup(self):
        """Setup of OM component."""
        self.add_input("machine_rating", 0.0, units="kW")
        self.add_input("tcc_per_kW", 0.0, units="USD/kW")
        self.add_input("offset_tcc_per_kW", 0.0, units="USD/kW")
        self.add_discrete_input("turbine_number", 0)
        self.add_output("tcc", 0.0, units="USD")

    def setup_partials(self):
        """Derivative setup for OM component."""
        # complex step for simple gradients
        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """Computation for the OM compoent."""
        # Unpack parameters
        t_rating = inputs["machine_rating"]
        n_turbine = discrete_inputs["turbine_number"]
        tcc_per_kW = inputs["tcc_per_kW"] + inputs["offset_tcc_per_kW"]
        outputs["tcc"] = n_turbine * tcc_per_kW * t_rating


class OperatingExpenses(om.ExplicitComponent):
    """
    A simple component to compute the operating costs.

    Inputs
    ------
    machine_rating : float
        rating of the wind turbine in kW
    opex_per_kW : float
        annual operating and maintenance costs per kW (as output from WISDEM
        tools)

    Discrete Inputs
    ---------------
    turbine_number : int
        number of turbines in the farm

    Outputs
    -------
    opex : float
        annual operating and maintenance costs in USD
    """

    def setup(self):
        """Setup of OM component."""
        self.add_input("machine_rating", 0.0, units="kW")
        self.add_input("opex_per_kW", 0.0, units="USD/kW/yr")
        self.add_discrete_input("turbine_number", 0)
        self.add_output("opex", 0.0, units="USD/yr")

    def setup_partials(self):
        """Derivative setup for OM component."""
        # complex step for simple gradients
        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """Computation for the OM compoent."""
        # Unpack parameters
        t_rating = inputs["machine_rating"]
        n_turbine = discrete_inputs["turbine_number"]
        opex_per_kW = inputs["opex_per_kW"]
        outputs["opex"] = n_turbine * opex_per_kW * t_rating


def LandBOSSE_setup_latents(modeling_options: dict) -> None:
    """
    A function to set up the LandBOSSE latent variables using modeling options.

    Parameters
    ----------
    prob : openmdao.api.Problem
        an OpenMDAO problem for which we want to setup the LandBOSSE latent
        variables
    modeling_options : dict
        a modeling options dictionary
    """

    # Define the mapping between OpenMDAO variable names and modeling_options keys
    offshore_fixed_keys = [
        "monopile_mass",
        "monopile_cost",
    ]

    offshore_floating_keys = [
        "num_mooring_lines",
        "mooring_line_mass",
        "mooring_line_diameter",
        "mooring_line_length",
        "anchor_mass",
        "floating_substructure_cost",
    ]

    if any(key in modeling_options["costs"] for key in offshore_fixed_keys):

        variable_mapping = {
            "num_turbines": modeling_options["layout"]["N_turbines"],
            "turbine_rating_MW": modeling_options["costs"]["rated_power"],
            "hub_height_meters": modeling_options["windIO_plant"]["wind_farm"][
                "turbine"
            ]["hub_height"],
            "rotor_diameter_m": modeling_options["windIO_plant"]["wind_farm"][
                "turbine"
            ]["rotor_diameter"],
            "number_of_blades": modeling_options["costs"]["num_blades"],
            "tower_mass": modeling_options["costs"]["tower_mass"],
            "nacelle_mass": modeling_options["costs"]["nacelle_mass"],
            "blade_mass": modeling_options["costs"]["blade_mass"],
            "commissioning_cost_kW": modeling_options["costs"]["commissioning_cost_kW"],
            "decommissioning_cost_kW": modeling_options["costs"][
                "decommissioning_cost_kW"
            ],
            # Offshore fixed-specific keys
            "monopile_mass": modeling_options["costs"]["monopile_mass"],
            "monopile_cost": modeling_options["costs"]["monopile_cost"],
        }

    elif any(key in modeling_options["costs"] for key in offshore_floating_keys):
        variable_mapping = {
            "num_turbines": modeling_options["layout"]["N_turbines"],
            "turbine_rating_MW": modeling_options["costs"]["rated_power"],
            "hub_height_meters": modeling_options["windIO_plant"]["wind_farm"][
                "turbine"
            ]["hub_height"],
            # "wind_shear_exponent": modeling_options["windIO_plant"]["site"]["energy_resource"]["wind_resource"]["shear"],
            "rotor_diameter_m": modeling_options["windIO_plant"]["wind_farm"][
                "turbine"
            ]["rotor_diameter"],
            "number_of_blades": modeling_options["costs"]["num_blades"],
            "tower_mass": modeling_options["costs"]["tower_mass"],
            "nacelle_mass": modeling_options["costs"]["nacelle_mass"],
            "blade_mass": modeling_options["costs"]["blade_mass"],
            "commissioning_cost_kW": modeling_options["costs"]["commissioning_cost_kW"],
            "decommissioning_cost_kW": modeling_options["costs"][
                "decommissioning_cost_kW"
            ],
            # Offshore floating-specific keys
            "num_mooring_lines": modeling_options["costs"]["num_mooring_lines"],
            "mooring_line_mass": modeling_options["costs"]["mooring_line_mass"],
            "mooring_line_diameter": modeling_options["costs"]["mooring_line_diameter"],
            "mooring_line_length": modeling_options["costs"]["mooring_line_length"],
            "anchor_mass": modeling_options["costs"]["anchor_mass"],
            "floating_substructure_cost": modeling_options["costs"][
                "floating_substructure_cost"
            ],
        }
    else:
        # this is the standard mapping for using LandBOSSE, since typically ORBIT should
        # be used for BOS costs for offshore systems.
        variable_mapping = {
            "num_turbines": modeling_options["layout"]["N_turbines"],
            "turbine_rating_MW": modeling_options["costs"]["rated_power"],
            "hub_height_meters": modeling_options["windIO_plant"]["wind_farm"][
                "turbine"
            ]["hub_height"],
            # "wind_shear_exponent": modeling_options["windIO_plant"]["site"][
            #     "energy_resource"
            # ]["wind_resource"].get("shear", None),
            "rotor_diameter_m": modeling_options["windIO_plant"]["wind_farm"][
                "turbine"
            ]["rotor_diameter"],
            "number_of_blades": modeling_options["costs"]["num_blades"],
            "rated_thrust_N": modeling_options["costs"]["rated_thrust_N"],
            "gust_velocity_m_per_s": modeling_options["costs"]["gust_velocity_m_per_s"],
            "blade_surface_area": modeling_options["costs"]["blade_surface_area"],
            "tower_mass": modeling_options["costs"]["tower_mass"],
            "nacelle_mass": modeling_options["costs"]["nacelle_mass"],
            "hub_mass": modeling_options["costs"]["hub_mass"],
            "blade_mass": modeling_options["costs"]["blade_mass"],
            "foundation_height": modeling_options["costs"]["foundation_height"],
            "commissioning_cost_kW": modeling_options["costs"]["commissioning_cost_kW"],
            "decommissioning_cost_kW": modeling_options["costs"][
                "decommissioning_cost_kW"
            ],
            "trench_len_to_substation_km": modeling_options["costs"][
                "trench_len_to_substation_km"
            ],
            "distance_to_interconnect_mi": modeling_options["costs"][
                "distance_to_interconnect_mi"
            ],
            "interconnect_voltage_kV": modeling_options["costs"][
                "interconnect_voltage_kV"
            ],
        }
    print(f"DEBUG!!!!! variable_mapping: {variable_mapping}")
    # set_values(prob, variable_map=variable_mapping)

    return variable_mapping


def ORBIT_setup_latents(modeling_options: dict) -> None:
    """
    A function to set up the ORBIT latent variables using modeling options.

    Parameters
    ----------
    prob : openmdao.api.Problem
        an OpenMDAO problem for which we want to setup the ORBIT latent
        variables
    modeling_options : dict
        a modeling options dictionary
    """

    # Define the mapping between OpenMDAO variable names and modeling_options keys
    variable_mapping = {
        "turbine_rating": modeling_options["costs"]["rated_power"],
        "site_depth": modeling_options["site_depth"],
        "number_of_turbines": modeling_options["layout"]["N_turbines"],
        "number_of_blades": modeling_options["costs"]["num_blades"],
        "hub_height": modeling_options["windIO_plant"]["wind_farm"]["turbine"][
            "hub_height"
        ],
        "turbine_rotor_diameter": modeling_options["windIO_plant"]["wind_farm"][
            "turbine"
        ]["rotor_diameter"],
        "tower_length": modeling_options["costs"]["tower_length"],
        "tower_mass": modeling_options["costs"]["tower_mass"],
        "nacelle_mass": modeling_options["costs"]["nacelle_mass"],
        "blade_mass": modeling_options["costs"]["blade_mass"],
        "turbine_capex": modeling_options["costs"]["turbine_capex"],
        "site_mean_windspeed": modeling_options["costs"]["site_mean_windspeed"],
        "turbine_rated_windspeed": modeling_options["costs"]["turbine_rated_windspeed"],
        "commissioning_cost_kW": modeling_options["costs"]["commissioning_cost_kW"],
        "decommissioning_cost_kW": modeling_options["costs"]["decommissioning_cost_kW"],
        "plant_substation_distance": modeling_options["costs"][
            "plant_substation_distance"
        ],
        "interconnection_distance": modeling_options["costs"][
            "interconnection_distance"
        ],
        "site_distance": modeling_options["costs"]["site_distance"],
        "site_distance_to_landfall": modeling_options["costs"][
            "site_distance_to_landfall"
        ],
        "port_cost_per_month": modeling_options["costs"]["port_cost_per_month"],
        "construction_insurance": modeling_options["costs"]["construction_insurance"],
        "construction_financing": modeling_options["costs"]["construction_financing"],
        "contingency": modeling_options["costs"]["contingency"],
        "site_auction_price": modeling_options["costs"]["site_auction_price"],
        "site_assessment_cost": modeling_options["costs"]["site_assessment_cost"],
        "construction_plan_cost": modeling_options["costs"]["construction_plan_cost"],
        "installation_plan_cost": modeling_options["costs"]["installation_plan_cost"],
        "boem_review_cost": modeling_options["costs"]["boem_review_cost"],
    }

    # Add floating-foundation specific keys if applicable
    if modeling_options["floating"]:
        variable_mapping.update(
            {
                "num_mooring_lines": modeling_options["costs"]["num_mooring_lines"],
                "mooring_line_mass": modeling_options["costs"]["mooring_line_mass"],
                "mooring_line_diameter": modeling_options["costs"][
                    "mooring_line_diameter"
                ],
                "mooring_line_length": modeling_options["costs"]["mooring_line_length"],
                "anchor_mass": modeling_options["costs"]["anchor_mass"],
                "transition_piece_mass": modeling_options["costs"][
                    "transition_piece_mass"
                ],
                "transition_piece_cost": modeling_options["costs"][
                    "transition_piece_cost"
                ],
                "floating_substructure_cost": modeling_options["costs"][
                    "floating_substructure_cost"
                ],
            }
        )
    # Add fixed-foundation (mooring) specific keys if applicable
    else:
        variable_mapping.update(
            {
                "monopile_mass": modeling_options["turbine"]["costs"]["monopile_mass"],
                "monopile_cost": modeling_options["turbine"]["costs"]["monopile_cost"],
                "monopile_length": modeling_options["turbine"]["geometry"][
                    "monopile_length"
                ],
                "monopile_diameter": modeling_options["turbine"]["geometry"][
                    "monopile_diameter"
                ],
                "transition_piece_mass": modeling_options["turbine"]["costs"][
                    "transition_piece_mass"
                ],
                "transition_piece_cost": modeling_options["turbine"]["costs"][
                    "transition_piece_cost"
                ],
            }
        )
    # TODO include jacket-type foundation
    # # if jacket
    # prob.set_val(
    #     comp2promotion_map["orbit.orbit.jacket_r_foot"],
    #     modeling_options["turbine"]["costs"]["jacket_r_foot"])
    # prob.set_val(
    #     comp2promotion_map["orbit.orbit.jacket_length"],
    #     modeling_options["turbine"]["costs"]["jacket_length"])
    # prob.set_val(
    #     comp2promotion_map["orbit.orbit.jacket_mass"],
    #     modeling_options["turbine"]["costs"]["jacket_mass"])
    # prob.set_val(
    #     comp2promotion_map["orbit.orbit.jacket_cost"],
    #     modeling_options["turbine"]["costs"]["jacket_cost"])
    # prob.set_val(
    #     comp2promotion_map["orbit.orbit.transition_piece_mass"],
    #     modeling_options["turbine"]["costs"]["transition_piece_mass"])
    # prob.set_val(
    #     comp2promotion_map["orbit.orbit.transition_piece_cost"],
    #     modeling_options["turbine"]["costs"]["transition_piece_cost"])

    # set_values(prob, variable_map=variable_mapping)
    return variable_mapping


def FinanceSE_setup_latents(modeling_options):
    """
    A function to set up the FinanceSE latent variables using modeling options.

    Parameters
    ----------
    prob : openmdao.api.Problem
        an OpenMDAO problem for which we want to setup the FinanceSE latent
        variables
    modeling_options : dict
        a modeling options dictionary
    """

    # Define the mapping between OpenMDAO variable names and modeling_options keys
    variable_mapping = {
        "turbine_number": int(modeling_options["layout"]["N_turbines"]),
        "machine_rating": modeling_options["costs"]["rated_power"] * 1.0e3,
        "tcc_per_kW": modeling_options["costs"]["tcc_per_kW"],
        "opex_per_kW": modeling_options["costs"]["opex_per_kW"],
    }

    # set_values(prob, variable_map=variable_mapping)

    return variable_mapping


def set_values(prob, variable_map: dict) -> None:
    """
    Set values in an OpenMDAO problem based on a mapping of variable names to values.

    This function dynamically maps core variable names to their promoted names in the
    OpenMDAO problem and sets their values using the provided `variable_map`.

    Parameters
    ----------
    prob : openmdao.api.Problem
        The OpenMDAO problem instance where the variables are to be set.
    variable_map : dict
        A dictionary mapping core variable names (keys) to their corresponding values
        (values) that need to be set in the OpenMDAO problem.

    Returns
    -------
    None
    """

    # # Get a map from the component variables to the promotion variables
    promotion_map = {
        v[0].split(".")[-1]: v[-1]["prom_name"]
        for v in prob.model.list_vars(val=False, out_stream=None)
    }

    # Iterate over the mapping and set values in the OpenMDAO problem
    for full_name in promotion_map:
        prom_name = promotion_map[full_name]
        core_name = prom_name.split(".")[-1]
        if core_name in promotion_map:
            try:
                prob.set_val(prom_name, variable_map[core_name])
            except:
                print(
                    f"{core_name} not provided in turbine input, using WISDEM default"
                )

    # # Get a map from the component variables to the promotion variables
    # promotion_map = {
    #     v[0]: v[-1]["prom_name"]
    #     for v in prob.model.list_vars(val=False, out_stream=None)
    # }

    # # Iterate over the mapping and set values in the OpenMDAO problem
    # for full_name in promotion_map:

    #     prom_name = promotion_map[full_name]
    #     core_name = prom_name.split(".")[-1]
    #     # if "turbine_number" in full_name:
    #     #     import pdb; pdb.set_trace()
    #     if core_name in variable_map:
    #         print(core_name)
    #         try:
    #             prob.set_val(full_name, variable_map[core_name])
    #         except:
    #             print(f"{core_name} not provided in Ard input, using WISDEM default")


# ['financese.machine_rating', 'opex.machine_rating', 'orbit.orbit.turbine_rating', 'tcc.machine_rating']
