from eppy import modeleditor
from eppy.modeleditor import IDF
from eppy.runner.run_functions import EnergyPlusRunError
import os
import shutil
import pandas as pd


#####################################################
# Separate expensive functions for various subcases #
#####################################################
# Climate files for Kuwait were downloaded from climate.onebuilding.org
# We have three climate variants:
# - desert:  Airport, as definitely dry desert climate
# - oasis:   Wafra, as local oasis with agricultural land
# - coastal: Nuwaiseeb, as humid coastal resort

def simulate_desert_villa(glazing_open_facade, shading_open_facade, glazing_closed_facade,
                          wwr_front, exterior_wall, insulation_thickness):
    return simulate_kw_villa(climate='desert',
                             orientation=180,
                             neighbor_left=True, neighbor_right=True, neighbor_back=True,
                             glazing_open_facade=glazing_open_facade,
                             shading_open_facade=shading_open_facade,
                             glazing_closed_facade=glazing_closed_facade,
                             wwr_front=wwr_front,
                             exterior_wall=exterior_wall,
                             insulation_thickness=insulation_thickness)


def simulate_oasis_villa(glazing_open_facade, shading_open_facade, glazing_closed_facade,
                         wwr_front, exterior_wall, insulation_thickness):
    return simulate_kw_villa(climate='oasis',
                             orientation=180,
                             neighbor_left=True, neighbor_right=True, neighbor_back=True,
                             glazing_open_facade=glazing_open_facade,
                             shading_open_facade=shading_open_facade,
                             glazing_closed_facade=glazing_closed_facade,
                             wwr_front=wwr_front,
                             exterior_wall=exterior_wall,
                             insulation_thickness=insulation_thickness)


def simulate_coastal_villa(glazing_open_facade, shading_open_facade, glazing_closed_facade,
                           wwr_front, exterior_wall, insulation_thickness):
    return simulate_kw_villa(climate='coastal',
                             orientation=180,
                             neighbor_left=True, neighbor_right=True, neighbor_back=True,
                             glazing_open_facade=glazing_open_facade,
                             shading_open_facade=shading_open_facade,
                             glazing_closed_facade=glazing_closed_facade,
                             wwr_front=wwr_front,
                             exterior_wall=exterior_wall,
                             insulation_thickness=insulation_thickness)


def simulate_coastal_villa_lux(glazing_open_facade, shading_open_facade, glazing_closed_facade,
                               wwr_front, wwr_back,
                               exterior_wall, insulation_thickness):
    return simulate_kw_villa(climate='coastal',
                             orientation=90,
                             neighbor_left=True, neighbor_right=True, neighbor_back=False,
                             glazing_open_facade=glazing_open_facade,
                             shading_open_facade=shading_open_facade,
                             glazing_closed_facade=glazing_closed_facade,
                             wwr_front=wwr_front,
                             wwr_back=wwr_back,
                             exterior_wall=exterior_wall,
                             insulation_thickness=insulation_thickness)


###################################################
# Common underlying expensive simulation function #
###################################################
delete_files = False         # whether to remove simulation outputs


def simulate_kw_villa_parallel(params):
    return simulate_kw_villa(*params)


def simulate_kw_villa(climate='desert',
                      orientation=180,
                      neighbor_left=True,
                      neighbor_right=True,
                      neighbor_back=True,
                      glazing_open_facade=2,
                      shading_open_facade='ext_shade',
                      glazing_closed_facade=1,
                      wwr_front=0.25,
                      wwr_left=0.10,
                      wwr_right=0.10,
                      wwr_back=0.10,
                      exterior_wall=1,
                      insulation_thickness=0.10):

    # create KW villa model
    idf = create_kw_villa(climate, orientation,
                          neighbor_left, neighbor_right, neighbor_back,
                          glazing_open_facade, shading_open_facade,
                          glazing_closed_facade,
                          wwr_front, wwr_left, wwr_right, wwr_back,
                          exterior_wall, insulation_thickness)

    # prepare the arguments for running Energyplus
    idf_version = idf.idfobjects['version'][0].Version_Identifier.split('.')
    idf_version.extend([0] * (3 - len(idf_version)))
    idf_version_str = '-'.join([str(item) for item in idf_version])

    file_name = idf.idfname
    # remove the extension from the filename to serve as the prefix for output files
    prefix = os.path.basename(file_name)
    dot_pos = prefix.rindex('.')
    prefix = prefix[0:dot_pos]

    args = {
        'ep_version': idf_version_str,  # runIDFs needs the version number
        'expandobjects': True,
        'epmacro': True,
        'readvars': True,
        'output_directory': os.path.join(os.path.dirname(file_name), 'exec', prefix),
        'output_prefix': prefix,
        'output_suffix': 'D',
        'verbose': 'q'
    }

    # make sure the output directory is not there
    shutil.rmtree(args['output_directory'], ignore_errors=True)
    # create the output directory
    os.mkdir(args['output_directory'])

    try:
        # run Energyplus
        print(f'Starting EnergyPlus simulation for {idf.idfname}')
        idf.run(**args)
        # print(f'Simulation for {idf.idfname} completed!')

        # readvars was set to True, so ReadVarsESO collected the meters in a separate csv file
        csv_filename = os.path.join(args['output_directory'], f'{prefix}-meter.csv')

        try:
            # read the csv file to obtain heating and cooling energies
            # what happens with lighting and photovoltaic energies?
            tmp_df = pd.read_csv(csv_filename)
            district_heating = tmp_df.iloc[2, 1]
            district_cooling = tmp_df.iloc[2, 2]

            if delete_files:
                shutil.rmtree(args['output_directory'], ignore_errors=True)

            return 3.613 * district_heating + 1.056 * district_cooling

            # PNNL fuel factor coefficients:
            # Electricity     = 3.167
            # Natural Gas     = 1.084
            # DistrictHeating = 3.613
            # DistrictCooling = 1.056

        except Exception:
            print(f'csv file: {csv_filename} not found!')
            return float('inf')

    except EnergyPlusRunError as e:
        print(f'Simulation run for {idf.idfname} failed: {str(e)}')
        return float('inf')


def create_kw_villa(climate='desert',
                    orientation=180,
                    neighbor_left=True,
                    neighbor_right=True,
                    neighbor_back=True,
                    glazing_open_facade=2,
                    shading_open_facade='ext_shade',
                    glazing_closed_facade=1,
                    wwr_front=0.25,
                    wwr_left=0.10,
                    wwr_right=0.10,
                    wwr_back=0.10,
                    exterior_wall=1,
                    insulation_thickness=0.1):
    """
    The method creates a variant of a Kuwaiti villa,
    inspired by houses recently built in West Abdullah Al Mubarak residential area.
    The front facade of the house is always open,
    while the remaining facades may be open or 'closed',
    depending on the presence of neighbors on the left, right and back side.
    The created variant is determined by the following parameters:

    :param climate:
        The climate file used for simulation:
        - 'desert' = dry desert, uses the climate file for the airport
        - 'coastal' = humid coastal resort, uses the climate file for Nuwaiseeb
        - 'oasis' = oasis with agricultural land, uses the climate file for Wafra
    :param orientation:
        Orientation of the front facade (0 = North, 180 = South). Default is 180.
    :param neighbor_left:
        Indicates the presence of a neighboring house on the left side
    :param neighbor_right:
        Indicates the presence of a neighboring house on the right side
    :param neighbor_back:
        Indicates the presence of neighboring houses on the back side
    :param glazing_open_facade:
        The glazing type used for open facades:
        - 1: clear 6mm, air gap 13mm, clear 6mm
        - 2: bronze 6mm, air gap 13mm, clear 6mm (default)
        - 3: Pyr B clear 6mm, air gap 13mm, clear 6mm
        - 4: LoE tint 6mm, air gap 13mm, clear 6mm
        - 5: clear 6mm, air gap 13mm, coated poly-66, air gap 13mm, clear 6mm
        - 6: bronze 6mm, air gap 13mm, coated poly-66, air gap 13mm, clear 6mm
    :param shading_open_facade:
        The shading system on the exterior side of glazing on open facades:
        - 'int_shade': interior shade
        - 'ext_shade': exterior roller shade (default)
        - 'ext_blind': exterior slatted blinds
        Note that the shading system on closed facades is always set to exterior roller shade
    :param glazing_closed_facade:
        The glazing type used for 'closed' facades,
        for which a neighboring house is placed at the distance of only 3m apart,
        effectively blocking that facade from most of the sunshine throughout the year.
        The value varies from 1 to 6, with the same constructions as above.
    :param wwr_front:
        Windows-to-wall ratio on the front facade.
        Acceptable values are from 0.0 to 1.0. Default is 0.25.
    :param wwr_left:
        Windows-to-wall ratio on the left facade. Default is 0.10.
    :param wwr_right:
        Windows-to-wall ratio on the right facade. Default is 0.10.
    :param wwr_back:
        Windows-to-wall ratio on the back facade. Default is 0.10.
    :param exterior_wall:
        The exterior wall type used for all facades:
        - 1: single wythe “white” block (default)
        - 2: sandwich double wythe concrete block with polystyrene board in between
        - 3: cement block with exterior polystyrene board covered with plaster
        - 4: cement block with exterior polystyrene board covered with mechanically installed stone
    :param insulation_thickness:
        Thickness of the insulation layer in the exterior wall type (default = 0.1m)
    :return:
        the idf object ready to be simulated with eppy

    NOTE:
    EnergyPlus will report errors if any two vertices are placed less than 0.01m apart.
    Hence windows-to-wall ratios that are extremely close to 1.0 will be truncated
    so that the window vertices are at least 0.01m apart from the wall vertices.
    """
    try:
        IDF.setiddname('Energy+.idd')
    except modeleditor.IDDAlreadySetError as e:
        pass

    # load the basic KW villa model with the appropriate climate file,
    # obtained by adding .epw extension to the climate type
    idf = IDF('KW_villa.idf', 'KW_climate_'+climate+'.epw')

    if neighbor_left:
        add_neighbor_left(idf)

    if neighbor_right:
        add_neighbor_right(idf)

    if neighbor_back:
        add_neighbor_back(idf)

    if neighbor_back and neighbor_left:
        add_neighbor_back_left(idf)

    if neighbor_back and neighbor_right:
        add_neighbor_back_right(idf)

    set_orientation(idf, orientation)

    set_glazing_type(idf, glazing_open_facade, shading_open_facade, glazing_closed_facade,
                     neighbor_left, neighbor_right, neighbor_back)

    set_wwr(idf, wwr_front, wwr_left, wwr_right, wwr_back)

    set_exterior_wall(idf, exterior_wall, insulation_thickness)

    # prepare a unique idf name, so that the result files are distinguishable
    idf.idfname = f'kw_villa_{climate}_{orientation}_{glazing_open_facade}_{shading_open_facade}_{glazing_closed_facade}_{wwr_front}_{wwr_left}_{wwr_right}_{wwr_back}_{exterior_wall}_{insulation_thickness}.idf'

    return idf


def set_orientation(idf, orientation):
    building = idf.idfobjects['BUILDING'][0]
    building.North_Axis = orientation - 180


def add_neighbor_left(idf):
    nlp1 = idf.newidfobject('SHADING:BUILDING:DETAILED')
    nlp1.Name = 'neighbor_left_part1'
    nlp1.Transmittance_Schedule_Name = 'Constant transmittance schedule'
    nlp1.Number_of_Vertices = 4
    nlp1.Vertex_1_Xcoordinate = -3
    nlp1.Vertex_2_Xcoordinate = -3
    nlp1.Vertex_3_Xcoordinate = -3
    nlp1.Vertex_4_Xcoordinate = -3
    nlp1.Vertex_1_Ycoordinate = 0
    nlp1.Vertex_2_Ycoordinate = 2
    nlp1.Vertex_3_Ycoordinate = 2
    nlp1.Vertex_4_Ycoordinate = 0
    nlp1.Vertex_1_Zcoordinate = 0
    nlp1.Vertex_2_Zcoordinate = 0
    nlp1.Vertex_3_Zcoordinate = 5
    nlp1.Vertex_4_Zcoordinate = 5

    nlp2 = idf.newidfobject('SHADING:BUILDING:DETAILED')
    nlp2.Name = 'neighbor_left_part2'
    nlp2.Transmittance_Schedule_Name = 'Constant transmittance schedule'
    nlp2.Number_of_Vertices = 4
    nlp2.Vertex_1_Xcoordinate = -3
    nlp2.Vertex_2_Xcoordinate = -3
    nlp2.Vertex_3_Xcoordinate = -3
    nlp2.Vertex_4_Xcoordinate = -3
    nlp2.Vertex_1_Ycoordinate = 2
    nlp2.Vertex_2_Ycoordinate = 12
    nlp2.Vertex_3_Ycoordinate = 12
    nlp2.Vertex_4_Ycoordinate = 2
    nlp2.Vertex_1_Zcoordinate = 0
    nlp2.Vertex_2_Zcoordinate = 0
    nlp2.Vertex_3_Zcoordinate = 13
    nlp2.Vertex_4_Zcoordinate = 13

    nlp3 = idf.newidfobject('SHADING:BUILDING:DETAILED')
    nlp3.Name = 'neighbor_left_part3'
    nlp3.Transmittance_Schedule_Name = 'Constant transmittance schedule'
    nlp3.Number_of_Vertices = 4
    nlp3.Vertex_1_Xcoordinate = -3
    nlp3.Vertex_2_Xcoordinate = -3
    nlp3.Vertex_3_Xcoordinate = -3
    nlp3.Vertex_4_Xcoordinate = -3
    nlp3.Vertex_1_Ycoordinate = 12
    nlp3.Vertex_2_Ycoordinate = 22
    nlp3.Vertex_3_Ycoordinate = 22
    nlp3.Vertex_4_Ycoordinate = 12
    nlp3.Vertex_1_Zcoordinate = 0
    nlp3.Vertex_2_Zcoordinate = 0
    nlp3.Vertex_3_Zcoordinate = 16
    nlp3.Vertex_4_Zcoordinate = 16


def add_neighbor_right(idf):
    nrp1 = idf.newidfobject('SHADING:BUILDING:DETAILED')
    nrp1.Name = 'neighbor_right_part1'
    nrp1.Transmittance_Schedule_Name = 'Constant transmittance schedule'
    nrp1.Number_of_Vertices = 4
    nrp1.Vertex_1_Xcoordinate = 21
    nrp1.Vertex_2_Xcoordinate = 21
    nrp1.Vertex_3_Xcoordinate = 21
    nrp1.Vertex_4_Xcoordinate = 21
    nrp1.Vertex_1_Ycoordinate = 0
    nrp1.Vertex_2_Ycoordinate = 2
    nrp1.Vertex_3_Ycoordinate = 2
    nrp1.Vertex_4_Ycoordinate = 0
    nrp1.Vertex_1_Zcoordinate = 0
    nrp1.Vertex_2_Zcoordinate = 0
    nrp1.Vertex_3_Zcoordinate = 5
    nrp1.Vertex_4_Zcoordinate = 5

    nrp2 = idf.newidfobject('SHADING:BUILDING:DETAILED')
    nrp2.Name = 'neighbor_right_part2'
    nrp2.Transmittance_Schedule_Name = 'Constant transmittance schedule'
    nrp2.Number_of_Vertices = 4
    nrp2.Vertex_1_Xcoordinate = 21
    nrp2.Vertex_2_Xcoordinate = 21
    nrp2.Vertex_3_Xcoordinate = 21
    nrp2.Vertex_4_Xcoordinate = 21
    nrp2.Vertex_1_Ycoordinate = 2
    nrp2.Vertex_2_Ycoordinate = 12
    nrp2.Vertex_3_Ycoordinate = 12
    nrp2.Vertex_4_Ycoordinate = 2
    nrp2.Vertex_1_Zcoordinate = 0
    nrp2.Vertex_2_Zcoordinate = 0
    nrp2.Vertex_3_Zcoordinate = 13
    nrp2.Vertex_4_Zcoordinate = 13

    nrp3 = idf.newidfobject('SHADING:BUILDING:DETAILED')
    nrp3.Name = 'neighbor_right_part3'
    nrp3.Transmittance_Schedule_Name = 'Constant transmittance schedule'
    nrp3.Number_of_Vertices = 4
    nrp3.Vertex_1_Xcoordinate = 21
    nrp3.Vertex_2_Xcoordinate = 21
    nrp3.Vertex_3_Xcoordinate = 21
    nrp3.Vertex_4_Xcoordinate = 21
    nrp3.Vertex_1_Ycoordinate = 12
    nrp3.Vertex_2_Ycoordinate = 22
    nrp3.Vertex_3_Ycoordinate = 22
    nrp3.Vertex_4_Ycoordinate = 12
    nrp3.Vertex_1_Zcoordinate = 0
    nrp3.Vertex_2_Zcoordinate = 0
    nrp3.Vertex_3_Zcoordinate = 16
    nrp3.Vertex_4_Zcoordinate = 16


def add_neighbor_back(idf):
    nb = idf.newidfobject('SHADING:BUILDING:DETAILED')
    nb.Name = 'neighbor_back'
    nb.Transmittance_Schedule_Name = 'Constant transmittance schedule'
    nb.Number_of_Vertices = 4
    nb.Vertex_1_Xcoordinate = 0
    nb.Vertex_2_Xcoordinate = 18
    nb.Vertex_3_Xcoordinate = 18
    nb.Vertex_4_Xcoordinate = 0
    nb.Vertex_1_Ycoordinate = 26
    nb.Vertex_2_Ycoordinate = 26
    nb.Vertex_3_Ycoordinate = 26
    nb.Vertex_4_Ycoordinate = 26
    nb.Vertex_1_Zcoordinate = 0
    nb.Vertex_2_Zcoordinate = 0
    nb.Vertex_3_Zcoordinate = 16
    nb.Vertex_4_Zcoordinate = 16


def add_neighbor_back_left(idf):
    nblp0 = idf.newidfobject('SHADING:BUILDING:DETAILED')
    nblp0.Name = 'neighbor_back_left_part0'
    nblp0.Transmittance_Schedule_Name = 'Constant transmittance schedule'
    nblp0.Number_of_Vertices = 4
    nblp0.Vertex_1_Xcoordinate = -21
    nblp0.Vertex_2_Xcoordinate = -3
    nblp0.Vertex_3_Xcoordinate = -3
    nblp0.Vertex_4_Xcoordinate = -21
    nblp0.Vertex_1_Ycoordinate = 26
    nblp0.Vertex_2_Ycoordinate = 26
    nblp0.Vertex_3_Ycoordinate = 26
    nblp0.Vertex_4_Ycoordinate = 26
    nblp0.Vertex_1_Zcoordinate = 0
    nblp0.Vertex_2_Zcoordinate = 0
    nblp0.Vertex_3_Zcoordinate = 16
    nblp0.Vertex_4_Zcoordinate = 16

    nblp1 = idf.newidfobject('SHADING:BUILDING:DETAILED')
    nblp1.Name = 'neighbor_back_left_part1'
    nblp1.Transmittance_Schedule_Name = 'Constant transmittance schedule'
    nblp1.Number_of_Vertices = 4
    nblp1.Vertex_1_Xcoordinate = -3
    nblp1.Vertex_2_Xcoordinate = -3
    nblp1.Vertex_3_Xcoordinate = -3
    nblp1.Vertex_4_Xcoordinate = -3
    nblp1.Vertex_1_Ycoordinate = 48
    nblp1.Vertex_2_Ycoordinate = 46
    nblp1.Vertex_3_Ycoordinate = 46
    nblp1.Vertex_4_Ycoordinate = 48
    nblp1.Vertex_1_Zcoordinate = 0
    nblp1.Vertex_2_Zcoordinate = 0
    nblp1.Vertex_3_Zcoordinate = 5
    nblp1.Vertex_4_Zcoordinate = 5

    nblp2 = idf.newidfobject('SHADING:BUILDING:DETAILED')
    nblp2.Name = 'neighbor_back_left_part2'
    nblp2.Transmittance_Schedule_Name = 'Constant transmittance schedule'
    nblp2.Number_of_Vertices = 4
    nblp2.Vertex_1_Xcoordinate = -3
    nblp2.Vertex_2_Xcoordinate = -3
    nblp2.Vertex_3_Xcoordinate = -3
    nblp2.Vertex_4_Xcoordinate = -3
    nblp2.Vertex_1_Ycoordinate = 46
    nblp2.Vertex_2_Ycoordinate = 36
    nblp2.Vertex_3_Ycoordinate = 36
    nblp2.Vertex_4_Ycoordinate = 46
    nblp2.Vertex_1_Zcoordinate = 0
    nblp2.Vertex_2_Zcoordinate = 0
    nblp2.Vertex_3_Zcoordinate = 13
    nblp2.Vertex_4_Zcoordinate = 13

    nblp3 = idf.newidfobject('SHADING:BUILDING:DETAILED')
    nblp3.Name = 'neighbor_back_left_part3'
    nblp3.Transmittance_Schedule_Name = 'Constant transmittance schedule'
    nblp3.Number_of_Vertices = 4
    nblp3.Vertex_1_Xcoordinate = -3
    nblp3.Vertex_2_Xcoordinate = -3
    nblp3.Vertex_3_Xcoordinate = -3
    nblp3.Vertex_4_Xcoordinate = -3
    nblp3.Vertex_1_Ycoordinate = 36
    nblp3.Vertex_2_Ycoordinate = 26
    nblp3.Vertex_3_Ycoordinate = 26
    nblp3.Vertex_4_Ycoordinate = 36
    nblp3.Vertex_1_Zcoordinate = 0
    nblp3.Vertex_2_Zcoordinate = 0
    nblp3.Vertex_3_Zcoordinate = 16
    nblp3.Vertex_4_Zcoordinate = 16


def add_neighbor_back_right(idf):
    nbrp0 = idf.newidfobject('SHADING:BUILDING:DETAILED')
    nbrp0.Name = 'neighbor_back_right_part0'
    nbrp0.Transmittance_Schedule_Name = 'Constant transmittance schedule'
    nbrp0.Number_of_Vertices = 4
    nbrp0.Vertex_1_Xcoordinate = 21
    nbrp0.Vertex_2_Xcoordinate = 39
    nbrp0.Vertex_3_Xcoordinate = 39
    nbrp0.Vertex_4_Xcoordinate = 21
    nbrp0.Vertex_1_Ycoordinate = 26
    nbrp0.Vertex_2_Ycoordinate = 26
    nbrp0.Vertex_3_Ycoordinate = 26
    nbrp0.Vertex_4_Ycoordinate = 26
    nbrp0.Vertex_1_Zcoordinate = 0
    nbrp0.Vertex_2_Zcoordinate = 0
    nbrp0.Vertex_3_Zcoordinate = 16
    nbrp0.Vertex_4_Zcoordinate = 16

    nbrp1 = idf.newidfobject('SHADING:BUILDING:DETAILED')
    nbrp1.Name = 'neighbor_back_right_part1'
    nbrp1.Transmittance_Schedule_Name = 'Constant transmittance schedule'
    nbrp1.Number_of_Vertices = 4
    nbrp1.Vertex_1_Xcoordinate = 21
    nbrp1.Vertex_2_Xcoordinate = 21
    nbrp1.Vertex_3_Xcoordinate = 21
    nbrp1.Vertex_4_Xcoordinate = 21
    nbrp1.Vertex_1_Ycoordinate = 48
    nbrp1.Vertex_2_Ycoordinate = 46
    nbrp1.Vertex_3_Ycoordinate = 46
    nbrp1.Vertex_4_Ycoordinate = 48
    nbrp1.Vertex_1_Zcoordinate = 0
    nbrp1.Vertex_2_Zcoordinate = 0
    nbrp1.Vertex_3_Zcoordinate = 5
    nbrp1.Vertex_4_Zcoordinate = 5

    nbrp2 = idf.newidfobject('SHADING:BUILDING:DETAILED')
    nbrp2.Name = 'neighbor_back_right_part2'
    nbrp2.Transmittance_Schedule_Name = 'Constant transmittance schedule'
    nbrp2.Number_of_Vertices = 4
    nbrp2.Vertex_1_Xcoordinate = 21
    nbrp2.Vertex_2_Xcoordinate = 21
    nbrp2.Vertex_3_Xcoordinate = 21
    nbrp2.Vertex_4_Xcoordinate = 21
    nbrp2.Vertex_1_Ycoordinate = 46
    nbrp2.Vertex_2_Ycoordinate = 36
    nbrp2.Vertex_3_Ycoordinate = 36
    nbrp2.Vertex_4_Ycoordinate = 46
    nbrp2.Vertex_1_Zcoordinate = 0
    nbrp2.Vertex_2_Zcoordinate = 0
    nbrp2.Vertex_3_Zcoordinate = 13
    nbrp2.Vertex_4_Zcoordinate = 13

    nbrp3 = idf.newidfobject('SHADING:BUILDING:DETAILED')
    nbrp3.Name = 'neighbor_back_right_part3'
    nbrp3.Transmittance_Schedule_Name = 'Constant transmittance schedule'
    nbrp3.Number_of_Vertices = 4
    nbrp3.Vertex_1_Xcoordinate = 21
    nbrp3.Vertex_2_Xcoordinate = 21
    nbrp3.Vertex_3_Xcoordinate = 21
    nbrp3.Vertex_4_Xcoordinate = 21
    nbrp3.Vertex_1_Ycoordinate = 36
    nbrp3.Vertex_2_Ycoordinate = 26
    nbrp3.Vertex_3_Ycoordinate = 26
    nbrp3.Vertex_4_Ycoordinate = 36
    nbrp3.Vertex_1_Zcoordinate = 0
    nbrp3.Vertex_2_Zcoordinate = 0
    nbrp3.Vertex_3_Zcoordinate = 16
    nbrp3.Vertex_4_Zcoordinate = 16


def set_glazing_type(idf, glazing_open_facade, shading_open_facade, glazing_closed_facade,
                     neighbor_left, neighbor_right, neighbor_back):
    # set up the proper name of glazing constructions for open and closed facades
    # (these do not contain shading - they will be later specified in windowshadingcontrols!)
    glaz_constr_open = 'Exterior Window ' + str(glazing_open_facade)
    glaz_constr_closed = 'Exterior Window ' + str(glazing_closed_facade)

    # classify windows into various facades
    windows = idf.idfobjects['FENESTRATIONSURFACE:DETAILED']
    win_front = [w for w in windows if 'front_' in w.Name]   # this includes top floor glass door!
    win_left =  [w for w in windows if 'left_' in w.Name]
    win_right = [w for w in windows if 'right_' in w.Name]
    win_back =  [w for w in windows if 'back_' in w.Name]

    # front facade is always open
    for w in win_front:
        w.Construction_Name = glaz_constr_open

    # left facade
    if neighbor_left:
        # closed
        for w in win_left:
            w.Construction_Name = glaz_constr_closed
    else:
        # open
        for w in win_left:
            w.Construction_Name = glaz_constr_open

    # right facade: open or closed?
    if neighbor_right:
        # closed
        for w in win_right:
            w.Construction_Name = glaz_constr_closed
    else:
        # open
        for w in win_right:
            w.Construction_Name = glaz_constr_open

    # back facade
    if neighbor_back:
        # closed
        for w in win_back:
            w.Construction_Name = glaz_constr_closed
    else:
        # open
        for w in win_back:
            w.Construction_Name = glaz_constr_open

    # firstly: one or two types of windowshadingcontrol?
    # if there are no neighbors from either side, then all facades are open, so one WSC
    # otherwise, check that open and closed facades have the same shading and glazing types
    if (not neighbor_left and not neighbor_right and not neighbor_back) or \
            (shading_open_facade == 'ext_shade' and glazing_open_facade == glazing_closed_facade):
        # both open and closed facades have the same glazing and shading
        wsc_ground = idf.newidfobject('WINDOWSHADINGCONTROL')
        wsc_ground.Name = 'open_closed_shading_ground_floor'
        wsc_ground.Zone_Name = 'ground_floor'

        wsc_first = idf.newidfobject('WINDOWSHADINGCONTROL')
        wsc_first.Name = 'open_closed_shading_first_floor'
        wsc_first.Zone_Name = 'first_floor'

        wsc_second = idf.newidfobject('WINDOWSHADINGCONTROL')
        wsc_second.Name = 'open_closed_shading_second_floor'
        wsc_second.Zone_Name = 'second_floor'

        wsc_top = idf.newidfobject('WINDOWSHADINGCONTROL')
        wsc_top.Name = 'open_closed_shading_top_floor'
        wsc_top.Zone_Name = 'top_floor'

        # common properties of these windowshadingcontrols
        for wsc in [wsc_ground, wsc_first, wsc_second, wsc_top]:
            wsc.Shading_Control_Sequence_Number = 1
            wsc.Shading_Type = 'ExteriorShade'
            wsc.Construction_with_Shading_Name = 'Exterior Window ' + str(glazing_open_facade) + ' Ext Shade'
            wsc.Shading_Control_Type = 'OnIfHighOutdoorAirTempAndHighHorizontalSolar'
            wsc.Setpoint = 30
            wsc.Glare_Control_Is_Active = 'No'
            if shading_open_facade == 'ext_blind':
                wsc.Type_of_Slat_Angle_Control_for_Blinds = 'BlockBeamSolar'
            wsc.Setpoint_2 = 250
            wsc.Multiple_Surface_Control_Type = 'Sequential'

        wsc_ground.Fenestration_Surface_1_Name = 'ground_floor_wall_front_window_left'
        wsc_ground.Fenestration_Surface_2_Name = 'ground_floor_wall_front_window_right'
        wsc_ground.Fenestration_Surface_3_Name = 'ground_floor_wall_left_window'
        wsc_ground.Fenestration_Surface_4_Name = 'ground_floor_wall_right_window'
        wsc_ground.Fenestration_Surface_5_Name = 'ground_floor_wall_back_window'

        wsc_first.Fenestration_Surface_1_Name = 'first_floor_wall_front_window'
        wsc_first.Fenestration_Surface_2_Name = 'first_floor_wall_left_window'
        wsc_first.Fenestration_Surface_3_Name = 'first_floor_wall_right_window'
        wsc_first.Fenestration_Surface_4_Name = 'first_floor_wall_back_window'

        wsc_second.Fenestration_Surface_1_Name = 'second_floor_wall_front_window'
        wsc_second.Fenestration_Surface_2_Name = 'second_floor_wall_left_window'
        wsc_second.Fenestration_Surface_3_Name = 'second_floor_wall_right_window'
        wsc_second.Fenestration_Surface_4_Name = 'second_floor_wall_back_window'

        wsc_top.Fenestration_Surface_1_Name = 'top_floor_wall_front_window_left'
        wsc_top.Fenestration_Surface_2_Name = 'top_floor_wall_front_window_right'
        wsc_top.Fenestration_Surface_3_Name = 'top_floor_wall_front_door'
        wsc_top.Fenestration_Surface_4_Name = 'top_floor_wall_left_window'
        wsc_top.Fenestration_Surface_5_Name = 'top_floor_wall_right_window'
        wsc_top.Fenestration_Surface_6_Name = 'top_floor_wall_back_window'
    else:
        # open facades either have different shading or different glazing than the closed facades,
        # so you need two different types of windowshadingcontrol

        # first, open facades...
        wsc_ground_open = idf.newidfobject('WINDOWSHADINGCONTROL')
        wsc_ground_open.Name = 'open_shading_ground_floor'
        wsc_ground_open.Zone_Name = 'ground_floor'

        wsc_first_open = idf.newidfobject('WINDOWSHADINGCONTROL')
        wsc_first_open.Name = 'open_shading_first_floor'
        wsc_first_open.Zone_Name = 'first_floor'

        wsc_second_open = idf.newidfobject('WINDOWSHADINGCONTROL')
        wsc_second_open.Name = 'open_shading_second_floor'
        wsc_second_open.Zone_Name = 'second_floor'

        wsc_top_open = idf.newidfobject('WINDOWSHADINGCONTROL')
        wsc_top_open.Name = 'open_shading_top_floor'
        wsc_top_open.Zone_Name = 'top_floor'

        # common properties of these windowshadingcontrols
        for wsc in [wsc_ground_open, wsc_first_open, wsc_second_open, wsc_top_open]:
            wsc.Shading_Control_Sequence_Number = 1
            if shading_open_facade == 'int_shade':
                wsc.Shading_Type = 'InteriorShade'
                wsc.Construction_with_Shading_Name = 'Exterior Window ' + str(glazing_open_facade) + ' Int Shade'
            elif shading_open_facade == 'ext_shade':
                wsc.Shading_Type = 'ExteriorShade'
                wsc.Construction_with_Shading_Name = 'Exterior Window ' + str(glazing_open_facade) + ' Ext Shade'
            else:    # shading_open_facade == 'ext_blind'
                wsc.Shading_Type = 'ExteriorBlind'
                wsc.Construction_with_Shading_Name = 'Exterior Window ' + str(glazing_open_facade) + ' Ext Blind'

            wsc.Shading_Control_Type = 'OnIfHighOutdoorAirTempAndHighHorizontalSolar'
            wsc.Setpoint = 30
            wsc.Glare_Control_Is_Active = 'No'
            if shading_open_facade == 'ext_blind':
                wsc.Type_of_Slat_Angle_Control_for_Blinds = 'BlockBeamSolar'
            wsc.Setpoint_2 = 250
            wsc.Multiple_Surface_Control_Type = 'Sequential'

        # now list windows on open facades
        wsc_ground_open.Fenestration_Surface_1_Name = 'ground_floor_wall_front_window_left'
        wsc_ground_open.Fenestration_Surface_2_Name = 'ground_floor_wall_front_window_right'
        wsc_first_open.Fenestration_Surface_1_Name = 'first_floor_wall_front_window'
        wsc_second_open.Fenestration_Surface_1_Name = 'second_floor_wall_front_window'
        wsc_top_open.Fenestration_Surface_1_Name = 'top_floor_wall_front_window_left'
        wsc_top_open.Fenestration_Surface_2_Name = 'top_floor_wall_front_window_right'
        wsc_top_open.Fenestration_Surface_3_Name = 'top_floor_wall_front_door'

        if not neighbor_left:
            # left facade is open
            wsc_ground_open.Fenestration_Surface_3_Name = 'ground_floor_wall_left_window'
            wsc_first_open.Fenestration_Surface_2_Name = 'first_floor_wall_left_window'
            wsc_second_open.Fenestration_Surface_2_Name = 'second_floor_wall_left_window'
            wsc_top_open.Fenestration_Surface_4_Name = 'top_floor_wall_left_window'

            if not neighbor_right:
                # right facade is open
                wsc_ground_open.Fenestration_Surface_4_Name = 'ground_floor_wall_right_window'
                wsc_first_open.Fenestration_Surface_3_Name = 'first_floor_wall_right_window'
                wsc_second_open.Fenestration_Surface_3_Name = 'second_floor_wall_right_window'
                wsc_top_open.Fenestration_Surface_5_Name = 'top_floor_wall_right_window'

                if not neighbor_back:
                    # back facade is open
                    wsc_ground_open.Fenestration_Surface_5_Name = 'ground_floor_wall_back_window'
                    wsc_first_open.Fenestration_Surface_4_Name = 'first_floor_wall_back_window'
                    wsc_second_open.Fenestration_Surface_4_Name = 'second_floor_wall_back_window'
                    wsc_top_open.Fenestration_Surface_6_Name = 'top_floor_wall_back_window'
                # else, back facade is closed
            else:
                # right facade is closed, so pass to back facade
                if not neighbor_back:
                    # back facade is open
                    wsc_ground_open.Fenestration_Surface_4_Name = 'ground_floor_wall_back_window'
                    wsc_first_open.Fenestration_Surface_3_Name = 'first_floor_wall_back_window'
                    wsc_second_open.Fenestration_Surface_3_Name = 'second_floor_wall_back_window'
                    wsc_top_open.Fenestration_Surface_5_Name = 'top_floor_wall_back_window'
                # else, back facade is closed
        else:
            # left facade is closed, so pass to right facade
            if not neighbor_right:
                # right facade is open
                wsc_ground_open.Fenestration_Surface_3_Name = 'ground_floor_wall_right_window'
                wsc_first_open.Fenestration_Surface_2_Name = 'first_floor_wall_right_window'
                wsc_second_open.Fenestration_Surface_2_Name = 'second_floor_wall_right_window'
                wsc_top_open.Fenestration_Surface_4_Name = 'top_floor_wall_right_window'

                if not neighbor_back:
                    # back facade is open
                    wsc_ground_open.Fenestration_Surface_4_Name = 'ground_floor_wall_back_window'
                    wsc_first_open.Fenestration_Surface_3_Name = 'first_floor_wall_back_window'
                    wsc_second_open.Fenestration_Surface_3_Name = 'second_floor_wall_back_window'
                    wsc_top_open.Fenestration_Surface_5_Name = 'top_floor_wall_back_window'
                # else, back facade is closed
            else:
                # right facade is closed, so pass to back facade
                if not neighbor_back:
                    # back facade is open
                    wsc_ground_open.Fenestration_Surface_3_Name = 'ground_floor_wall_back_window'
                    wsc_first_open.Fenestration_Surface_2_Name = 'first_floor_wall_back_window'
                    wsc_second_open.Fenestration_Surface_2_Name = 'second_floor_wall_back_window'
                    wsc_top_open.Fenestration_Surface_4_Name = 'top_floor_wall_back_window'
                # else, back facade is closed

        # secondly, closed facades!
        wsc_ground_closed = idf.newidfobject('WINDOWSHADINGCONTROL')
        wsc_ground_closed.Name = 'closed_shading_ground_floor'
        wsc_ground_closed.Zone_Name = 'ground_floor'

        wsc_first_closed = idf.newidfobject('WINDOWSHADINGCONTROL')
        wsc_first_closed.Name = 'closed_shading_first_floor'
        wsc_first_closed.Zone_Name = 'first_floor'

        wsc_second_closed = idf.newidfobject('WINDOWSHADINGCONTROL')
        wsc_second_closed.Name = 'closed_shading_second_floor'
        wsc_second_closed.Zone_Name = 'second_floor'

        wsc_top_closed = idf.newidfobject('WINDOWSHADINGCONTROL')
        wsc_top_closed.Name = 'closed_shading_top_floor'
        wsc_top_closed.Zone_Name = 'top_floor'

        # common properties of these windowshadingcontrols
        for wsc in [wsc_ground_closed, wsc_first_closed, wsc_second_closed, wsc_top_closed]:
            wsc.Shading_Control_Sequence_Number = 2
            wsc.Shading_Type = 'ExteriorShade'
            wsc.Construction_with_Shading_Name = 'Exterior Window ' + str(glazing_closed_facade) + ' Ext Shade'
            wsc.Shading_Control_Type = 'OnIfHighOutdoorAirTempAndHighHorizontalSolar'
            wsc.Setpoint = 30
            wsc.Glare_Control_Is_Active = 'No'
            wsc.Setpoint_2 = 250
            wsc.Multiple_Surface_Control_Type = 'Sequential'

        # now list windows on closed facades
        if neighbor_left:
            # left facade is closed
            wsc_ground_closed.Fenestration_Surface_1_Name = 'ground_floor_wall_left_window'
            wsc_first_closed.Fenestration_Surface_1_Name = 'first_floor_wall_left_window'
            wsc_second_closed.Fenestration_Surface_1_Name = 'second_floor_wall_left_window'
            wsc_top_closed.Fenestration_Surface_1_Name = 'top_floor_wall_left_window'

            if neighbor_right:
                # right facade is closed
                wsc_ground_closed.Fenestration_Surface_2_Name = 'ground_floor_wall_right_window'
                wsc_first_closed.Fenestration_Surface_2_Name = 'first_floor_wall_right_window'
                wsc_second_closed.Fenestration_Surface_2_Name = 'second_floor_wall_right_window'
                wsc_top_closed.Fenestration_Surface_2_Name = 'top_floor_wall_right_window'

                if neighbor_back:
                    # back facade is closed
                    wsc_ground_closed.Fenestration_Surface_3_Name = 'ground_floor_wall_back_window'
                    wsc_first_closed.Fenestration_Surface_3_Name = 'first_floor_wall_back_window'
                    wsc_second_closed.Fenestration_Surface_3_Name = 'second_floor_wall_back_window'
                    wsc_top_closed.Fenestration_Surface_3_Name = 'top_floor_wall_back_window'
                # else, back facade is open
            else:
                # right facade is open, so pass to back facade
                if neighbor_back:
                    # back facade is closed
                    wsc_ground_closed.Fenestration_Surface_2_Name = 'ground_floor_wall_back_window'
                    wsc_first_closed.Fenestration_Surface_2_Name = 'first_floor_wall_back_window'
                    wsc_second_closed.Fenestration_Surface_2_Name = 'second_floor_wall_back_window'
                    wsc_top_closed.Fenestration_Surface_2_Name = 'top_floor_wall_back_window'
                # else, back facade is open
        else:
            # left facade is open, so pass to right facade
            if neighbor_right:
                # right facade is closed
                wsc_ground_closed.Fenestration_Surface_1_Name = 'ground_floor_wall_right_window'
                wsc_first_closed.Fenestration_Surface_1_Name = 'first_floor_wall_right_window'
                wsc_second_closed.Fenestration_Surface_1_Name = 'second_floor_wall_right_window'
                wsc_top_closed.Fenestration_Surface_1_Name = 'top_floor_wall_right_window'

                if neighbor_back:
                    # back facade is closed
                    wsc_ground_closed.Fenestration_Surface_2_Name = 'ground_floor_wall_back_window'
                    wsc_first_closed.Fenestration_Surface_2_Name = 'first_floor_wall_back_window'
                    wsc_second_closed.Fenestration_Surface_2_Name = 'second_floor_wall_back_window'
                    wsc_top_closed.Fenestration_Surface_2_Name = 'top_floor_wall_back_window'
                # else, back facade is open
            else:
                # right facade is open, so pass to back facade
                if neighbor_back:
                    # back facade is closed
                    wsc_ground_closed.Fenestration_Surface_1_Name = 'ground_floor_wall_back_window'
                    wsc_first_closed.Fenestration_Surface_1_Name = 'first_floor_wall_back_window'
                    wsc_second_closed.Fenestration_Surface_1_Name = 'second_floor_wall_back_window'
                    wsc_top_closed.Fenestration_Surface_1_Name = 'top_floor_wall_back_window'
                # else, back facade is open


def set_wwr(idf, wwr_front, wwr_left, wwr_right, wwr_back):
    # ground floor and top floor have two front windows each,
    # due to a door appearing in them,
    # so they have to be treated separately
    windows = idf.idfobjects['FENESTRATIONSURFACE:DETAILED']
    win_front_double = [w for w in windows
                        if ('ground' in w.Name or 'top' in w.Name) and 'front_window' in w.Name]
    win_front_single = [w for w in windows
                        if ('first' in w.Name or 'second' in w.Name) and 'front_window' in w.Name]
    win_left = [w for w in windows if 'left_window' in w.Name]
    win_right = [w for w in windows if 'right_window' in w.Name]
    win_back = [w for w in windows if 'back_window' in w.Name]

    set_double_wwr(idf, win_front_double, wwr_front)

    set_single_wwr(idf, win_front_single, wwr_front)
    set_single_wwr(idf, win_left, wwr_left)
    set_single_wwr(idf, win_right, wwr_right)
    set_single_wwr(idf, win_back, wwr_back)


def set_double_wwr(idf, windows, wwr, minsep=0.02):
    r = wwr ** 0.5

    for w in windows:
        if w.Name == 'ground_floor_wall_front_window_left':
            boundaries = [[0, 0, 0], [8, 0, 0], [8, 0, 4], [0, 0, 4]]
            center = [4, 0, 0.8]
        elif w.Name == 'ground_floor_wall_front_window_right':
            boundaries = [[10, 0, 0], [18, 0, 0], [18, 0, 4], [10, 0, 4]]
            center = [14, 0, 0.8]
        elif w.Name == 'top_floor_wall_front_window_left':
            boundaries = [[0, 12, 12], [8, 12, 12], [8, 12, 16], [0, 12, 16]]
            center = [4, 12, 12.8]
        else:   # 'top_floor_wall_front_window_right'
            boundaries = [[10, 12, 12], [18, 12, 12], [18, 12, 16], [10, 12, 16]]
            center = [14, 12, 12.8]

        # compute new coordinates
        set_coordinates_with_ratio(w, r, boundaries, center)

        # are the window dimensions too small?
        width = ((w.Vertex_1_Xcoordinate - w.Vertex_3_Xcoordinate)**2 +
                 (w.Vertex_1_Ycoordinate - w.Vertex_3_Ycoordinate)**2) ** 0.5
        height = abs(w.Vertex_1_Zcoordinate - w.Vertex_3_Zcoordinate)

        if width < minsep or height < minsep:
            idf.removeidfobject(w)

        # are the window vertices too close to the boundary vertices?
        dist1 = ((w.Vertex_1_Xcoordinate - boundaries[0][0])**2 +
                 (w.Vertex_1_Ycoordinate - boundaries[0][1])**2 +
                 (w.Vertex_1_Zcoordinate - boundaries[0][2])**2) ** 0.5

        dist3 = ((w.Vertex_3_Xcoordinate - boundaries[2][0])**2 +
                 (w.Vertex_3_Ycoordinate - boundaries[2][1])**2 +
                 (w.Vertex_3_Zcoordinate - boundaries[2][2])**2) ** 0.5

        if dist1 < minsep or dist3 < minsep:
            total1 = ((boundaries[0][0] - center[0])**2 +
                      (boundaries[0][1] - center[1])**2 +
                      (boundaries[0][2] - center[2])**2) ** 0.5

            total3 = ((boundaries[2][0] - center[0])**2 +
                      (boundaries[2][1] - center[1])**2 +
                      (boundaries[2][2] - center[2])**2) ** 0.5

            newr = min((total1 - minsep)/total1, (total3 - minsep)/total3)
            set_coordinates_with_ratio(w, newr, boundaries, center)


def set_single_wwr(idf, windows, wwr, minsep=0.02):
    r = wwr ** 0.5
    walls = idf.idfobjects['BUILDINGSURFACE:DETAILED']

    for win in windows:
        # find the underlying wall
        wall = [wall for wall in walls if wall.Name==win.Building_Surface_Name][0]

        boundaries = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        boundaries[0][0] = wall.Vertex_1_Xcoordinate
        boundaries[0][1] = wall.Vertex_1_Ycoordinate
        boundaries[0][2] = wall.Vertex_1_Zcoordinate

        boundaries[1][0] = wall.Vertex_2_Xcoordinate
        boundaries[1][1] = wall.Vertex_2_Ycoordinate
        boundaries[1][2] = wall.Vertex_2_Zcoordinate

        boundaries[2][0] = wall.Vertex_3_Xcoordinate
        boundaries[2][1] = wall.Vertex_3_Ycoordinate
        boundaries[2][2] = wall.Vertex_3_Zcoordinate

        boundaries[3][0] = wall.Vertex_4_Xcoordinate
        boundaries[3][1] = wall.Vertex_4_Ycoordinate
        boundaries[3][2] = wall.Vertex_4_Zcoordinate

        center = [0, 0, 0]
        center[0] = (wall.Vertex_1_Xcoordinate + wall.Vertex_3_Xcoordinate) / 2
        center[1] = (wall.Vertex_1_Ycoordinate + wall.Vertex_3_Ycoordinate) / 2
        center[2] = 0.8 + min(wall.Vertex_1_Zcoordinate, wall.Vertex_3_Zcoordinate)

        # compute new coordinates
        set_coordinates_with_ratio(win, r, boundaries, center)

        # are the window dimensions too small?
        width = ((win.Vertex_1_Xcoordinate - win.Vertex_3_Xcoordinate)**2 +
                 (win.Vertex_1_Ycoordinate - win.Vertex_3_Ycoordinate)**2) ** 0.5
        height = abs(win.Vertex_1_Zcoordinate - win.Vertex_3_Zcoordinate)

        if width < minsep or height < minsep:
            idf.removeidfobject(win)

        # are the window vertices too close to the wall vertices?
        dist1 = ((win.Vertex_1_Xcoordinate - boundaries[0][0])**2 +
                 (win.Vertex_1_Ycoordinate - boundaries[0][1])**2 +
                 (win.Vertex_1_Zcoordinate - boundaries[0][2])**2) ** 0.5

        dist3 = ((win.Vertex_3_Xcoordinate - boundaries[2][0])**2 +
                 (win.Vertex_3_Ycoordinate - boundaries[2][1])**2 +
                 (win.Vertex_3_Zcoordinate - boundaries[2][2])**2) ** 0.5

        if dist1 < minsep or dist3 < minsep:
            total1 = ((wall.Vertex_1_Xcoordinate - center[0])**2 +
                      (wall.Vertex_1_Ycoordinate - center[1])**2 +
                      (wall.Vertex_1_Zcoordinate - center[2])**2) ** 0.5

            total3 = ((wall.Vertex_3_Xcoordinate - center[0])**2 +
                      (wall.Vertex_3_Ycoordinate - center[1])**2 +
                      (wall.Vertex_3_Zcoordinate - center[2])**2) ** 0.5

            newr = min((total1 - minsep)/total1, (total3 - minsep)/total3)
            set_coordinates_with_ratio(win, newr, boundaries, center)


def set_coordinates_with_ratio(w, r, boundaries, center):
    w.Vertex_1_Xcoordinate = r * boundaries[0][0] + (1 - r) * center[0]
    w.Vertex_2_Xcoordinate = r * boundaries[1][0] + (1 - r) * center[0]
    w.Vertex_3_Xcoordinate = r * boundaries[2][0] + (1 - r) * center[0]
    w.Vertex_4_Xcoordinate = r * boundaries[3][0] + (1 - r) * center[0]

    w.Vertex_1_Ycoordinate = r * boundaries[0][1] + (1 - r) * center[1]
    w.Vertex_2_Ycoordinate = r * boundaries[1][1] + (1 - r) * center[1]
    w.Vertex_3_Ycoordinate = r * boundaries[2][1] + (1 - r) * center[1]
    w.Vertex_4_Ycoordinate = r * boundaries[3][1] + (1 - r) * center[1]

    w.Vertex_1_Zcoordinate = r * boundaries[0][2] + (1 - r) * center[2]
    w.Vertex_2_Zcoordinate = r * boundaries[1][2] + (1 - r) * center[2]
    w.Vertex_3_Zcoordinate = r * boundaries[2][2] + (1 - r) * center[2]
    w.Vertex_4_Zcoordinate = r * boundaries[3][2] + (1 - r) * center[2]


def set_exterior_wall(idf, exterior_wall, insulation_thickness):
    if insulation_thickness >= 0.01:
        materials = idf.idfobjects['MATERIAL']
        insulation_material = [m for m in materials
                               if m.Name == 'Insulation: Expanded polystyrene with a given thickness'][0]
        insulation_material.Thickness = insulation_thickness

        walls = idf.idfobjects['BUILDINGSURFACE:DETAILED']
        walls = [w for w in walls if 'wall_' in w.Name]
        for w in walls:
            w.Construction_Name = 'Exterior Wall ' + str(exterior_wall)
    else: # treat insulation_thickness as 0.0
        walls = idf.idfobjects['BUILDINGSURFACE:DETAILED']
        walls = [w for w in walls if 'wall_' in w.Name]
        for w in walls:
            w.Construction_Name = 'Exterior Wall ' + str(exterior_wall) + ' without insulation'



# for testing purposes...
if __name__=="__main__":
    try:
        IDF.setiddname('Energy+.idd')
    except modeleditor.IDDAlreadySetError as e:
        pass

    # now simulate a variant or two :)

    # glazing_open_facade: {1,2,3,4,5,6}
    # shading_open_facade: {'int_shade', 'ext_shade', 'ext_blind'}
    # glazing_closed_facade: {1,2,3,4,5,6}
    # wwr_front: {0.1, 0.2, 0.3, ..., 0.9}
    # exterior_wall: {1,2,3,4}
    # insulation_thickness: {0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50}
    # energy = simulate_desert_villa(1, 'int_shade', 1, 0.4, 1, 0.1)
    # print(f'energy needed: {energy}')

    energy = simulate_kw_villa(climate='desert',
                               orientation=180,
                               neighbor_left=True,
                               neighbor_right=True,
                               neighbor_back=True,
                               glazing_open_facade=3,
                               shading_open_facade='ext_shade',
                               glazing_closed_facade=6,
                               wwr_front=0.4,
                               wwr_left=0.1,
                               wwr_right=0.1,
                               wwr_back=0.1,
                               exterior_wall=2,
                               insulation_thickness=0.50)

    # energy = simulate_coastal_villa_lux(glazing_open_facade=5,
    #                                     shading_open_facade='ext_blind',
    #                                     glazing_closed_facade=6,
    #                                     wwr_front=0.5,
    #                                     wwr_back=0.25,
    #                                     exterior_wall=4,
    #                                     insulation_thickness=0.15)
    print(f'energy needed: {energy}')

