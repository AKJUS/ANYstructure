from scipy.special import gammaln
from scipy.stats import gamma as gammadist
import numpy as np
import ANYstructure_local.helper as hlp
import os, time, datetime, json, random, math
import ANYstructure_local.SN_curve_parameters as snc

class Structure():
    '''
    Setting the properties for the plate and the stiffener. Takes a dictionary as argument.
    '''
    def __init__(self, main_dict, *args, **kwargs):
        super(Structure,self).__init__()
        self.main_dict = main_dict
        self.plate_th = main_dict['plate_thk'][0]
        self.web_height = main_dict['stf_web_height'][0]
        self.web_th = main_dict['stf_web_thk'][0]
        self.flange_width = main_dict['stf_flange_width'][0]
        self.flange_th = main_dict['stf_flange_thk'][0]
        self.mat_yield = main_dict['mat_yield'][0]
        self.mat_factor = main_dict['mat_factor'][0]
        self.span = main_dict['span'][0]
        self.spacing = main_dict['spacing'][0]
        self.structure_type = main_dict['structure_type'][0]
        self.sigma_y1=main_dict['sigma_y1'][0]
        self.sigma_y2=main_dict['sigma_y2'][0]
        self.sigma_x=main_dict['sigma_x'][0]
        self.tauxy=main_dict['tau_xy'][0]
        self.plate_kpp = main_dict['plate_kpp'][0]
        self.stf_kps = main_dict['stf_kps'][0]
        self.km1 = main_dict['stf_km1'][0]
        self.km2 = main_dict['stf_km2'][0]
        self.km3 = main_dict['stf_km3'][0]
        self.stiffener_type=main_dict['stf_type'][0]
        self.structure_types = main_dict['structure_types'][0]
        self.dynamic_variable_orientation = None
        if self.structure_type in self.structure_types['vertical']:
            self.dynamic_variable_orientation = 'z - vertical'
        elif self.structure_type in self.structure_types['horizontal']:
            self.dynamic_variable_orientation = 'x - horizontal'
        self._puls_method = main_dict['puls buckling method'][0]
        self._puls_boundary = main_dict['puls boundary'][0]
        self._puls_stf_end = main_dict['puls stiffener end'][0]
        self._puls_sp_or_up = main_dict['puls sp or up'][0]
        self._puls_up_boundary = main_dict['puls up boundary'][0]

        self._zstar_optimization = main_dict['zstar_optimization'][0]
        try:
            self.girder_lg=main_dict['girder_lg'][0]
        except KeyError:
            self.girder_lg = 10
        try:
            self.pressure_side = main_dict['press_side'][0]
        except KeyError:
            self.pressure_side = 'p'

    def __str__(self):
        '''
        Returning all properties.
        '''
        return \
            str(
            '\n Plate field span:              ' + str(round(self.span,3)) + ' meters' +
            '\n Stiffener spacing:             ' + str(self.spacing*1000)+' mm'+
            '\n Plate thickness:               ' + str(self.plate_th*1000)+' mm'+
            '\n Stiffener web height:          ' + str(self.web_height*1000)+' mm'+
            '\n Stiffener web thickness:       ' + str(self.web_th*1000)+' mm'+
            '\n Stiffener flange width:        ' + str(self.flange_width*1000)+' mm'+
            '\n Stiffener flange thickness:    ' + str(self.flange_th*1000)+' mm'+
            '\n Material yield:                ' + str(self.mat_yield/1e6)+' MPa'+
            '\n Structure/stiffener type:      ' + str(self.structure_type)+'/'+(self.stiffener_type)+
            '\n Dynamic load varible_          ' + str(self.dynamic_variable_orientation)+
            '\n Plate fixation paramter,kpp:   ' + str(self.plate_kpp) + ' ' +
            '\n Stf. fixation paramter,kps:    ' + str(self.stf_kps) + ' ' +
            '\n Global stress, sig_y1/sig_y2:  ' + str(round(self.sigma_y1,3))+'/'+str(round(self.sigma_y2,3))+ ' MPa' +
            '\n Global stress, sig_x:          ' + str(round(self.sigma_x,3)) + ' MPa' +
            '\n Global shear, tau_xy:          ' + str(round(self.tauxy,3)) + ' MPa' +
            '\n km1,km2,km3:                   ' + str(self.km1)+'/'+str(self.km2)+'/'+str(self.km3)+
            '\n Pressure side (p-plate/s-stf): ' + str(self.pressure_side) + ' ')

    def get_beam_string(self):
        ''' Returning a string. '''
        base_name = self.stiffener_type+ '_' + str(round(self.web_height*1000, 0)) + 'x' + \
                   str(round(self.web_th*1000, 0))
        if self.stiffener_type == 'FB':
            ret_str = base_name
        else:
            ret_str = base_name + '__' + str(round(self.flange_width*1000, 0)) + 'x' + \
                      str(round(self.flange_th*1000, 0))

        ret_str = ret_str.replace('.', '_')

        return ret_str

    def get_structure_types(self):
        return self.structure_types

    def get_z_opt(self):
        return self._zstar_optimization

    def get_puls_method(self):
        return self._puls_method

    def get_puls_boundary(self):
        return self._puls_boundary

    def get_puls_stf_end(self):
        return self._puls_stf_end

    def get_puls_sp_or_up(self):
        return self._puls_sp_or_up

    def get_puls_up_boundary(self):
        return self._puls_up_boundary

    def get_one_line_string(self):
        ''' Returning a one line string. '''
        return 'pl_'+str(round(self.spacing*1000, 1))+'x'+str(round(self.plate_th*1000,1))+' stf_'+self.stiffener_type+\
               str(round(self.web_height*1000,1))+'x'+str(round(self.web_th*1000,1))+'+'\
               +str(round(self.flange_width*1000,1))+'x'+\
               str(round(self.flange_th*1000,1))

    def get_report_stresses(self):
        'Return the stresses to the report'
        return 'sigma_y1: '+str(round(self.sigma_y1,1))+' sigma_y2: '+str(round(self.sigma_y2,1))+ \
               ' sigma_x: ' + str(round(self.sigma_x,1))+' tauxy: '+ str(round(self.tauxy,1))

    def get_extended_string(self):
        ''' Some more information returned. '''
        return 'span: '+str(round(self.span,4))+' structure type: '+ self.structure_type + ' stf. type: ' + \
               self.stiffener_type + ' pressure side: ' + self.pressure_side
    
    def get_sigma_y1(self):
        '''
        Return sigma_y1
        :return:
        '''
        return self.sigma_y1
    def get_sigma_y2(self):
        '''
        Return sigma_y2
        :return:
        '''
        return self.sigma_y2
    def get_sigma_x(self):
        '''
        Return sigma_x
        :return:
        '''
        return self.sigma_x
    def get_tau_xy(self):
        '''
        Return tau_xy
        :return:
        '''
        return self.tauxy
    def get_s(self):
        '''
        Return the spacing
        :return:
        '''
        return self.spacing
    def get_pl_thk(self):
        '''
        Return the plate thickness
        :return:
        '''
        return self.plate_th
    def get_web_h(self):
        '''
        Return the web heigh
        :return:
        '''
        return self.web_height
    def get_web_thk(self):
        '''
        Return the spacing
        :return:
        '''
        return self.web_th
    def get_fl_w(self):
        '''
        Return the flange width
        :return:
        '''
        return self.flange_width
    def get_fl_thk(self):
        '''
        Return the flange thickness
        :return:
        '''
        return self.flange_th
    def get_fy(self):
        '''
        Return material yield
        :return:
        '''
        return self.mat_yield
    def get_mat_factor(self):
        return self.mat_factor
    def get_span(self):
        '''
        Return the span
        :return:
        '''
        return self.span
    def get_lg(self):
        '''
        Return the girder length
        :return:
        '''
        return self.girder_lg
    def get_kpp(self):
        '''
        Return var
        :return:
        '''
        return self.plate_kpp
    def get_kps(self):
        '''
        Return var
        :return:
        '''
        return self.stf_kps
    def get_km1(self):
        '''
        Return var
        :return:
        '''
        return self.km1
    def get_km2(self):
        '''
        Return var
        :return:
        '''
        return self.km2
    def get_km3(self):
        '''
        Return var
        :return:
        '''
        return self.km3
    def get_side(self):
        '''
        Return the checked pressure side.
        :return: 
        '''
        return self.pressure_side
    def get_tuple(self):
        ''' Return a tuple of the plate stiffener'''
        return (self.spacing, self.plate_th, self.web_height, self.web_th, self.flange_width,
                self.flange_th, self.span, self.girder_lg, self.stiffener_type)

    def get_section_modulus(self, efficient_se = None, dnv_table = False):
        '''
        Returns the section modulus.
        :param efficient_se: 
        :return: 
        '''
        #Plate. When using DNV table, default values are used for the plate
        b1 = self.spacing if efficient_se==None else efficient_se
        tf1 = self.plate_th

        #Stiffener
        tf2 = self.flange_th
        b2 = self.flange_width
        h = self.flange_th+self.web_height+self.plate_th
        tw = self.web_th
        hw = self.web_height

        # cross section area
        Ax = tf1 * b1 + tf2 * b2 + hw * tw

        assert Ax != 0, 'Ax cannot be 0'
        # distance to center of gravity in z-direction
        ez = (tf1 * b1 * tf1 / 2 + hw * tw * (tf1 + hw / 2) + tf2 * b2 * (tf1 + hw + tf2 / 2)) / Ax

        #ez = (tf1 * b1 * (h - tf1 / 2) + hw * tw * (tf2 + hw / 2) + tf2 * b2 * (tf2 / 2)) / Ax
        # moment of inertia in y-direction (c is centroid)

        Iyc = (1 / 12) * (b1 * math.pow(tf1, 3) + b2 * math.pow(tf2, 3) + tw * math.pow(hw, 3))
        Iy = Iyc + (tf1 * b1 * math.pow(tf1 / 2, 2) + tw * hw * math.pow(tf1+hw / 2, 2) +
             tf2 * b2 * math.pow(tf1+hw+tf2 / 2, 2)) - Ax * math.pow(ez, 2)

        # elastic section moduluses y-axis
        Wey1 = Iy / (h - ez)
        Wey2 = Iy / ez
        return Wey1, Wey2
    def get_plasic_section_modulus(self):
        '''
        Returns the plastic section modulus
        :return:
        '''
        tf1 = self.plate_th
        tf2 = self.flange_th
        b1 = self.spacing
        b2 = self.flange_width
        h = self.flange_th+self.web_height+self.plate_th
        tw = self.web_th
        hw = self.web_height

        Ax = tf1 * b1 + tf2 * b2 + (h-tf1-tf2) * tw

        ezpl = (Ax/2-b1*tf1)/tw+tf1

        az1 = h-ezpl-tf1
        az2 = ezpl-tf2

        Wy1 = b1*tf1*(az1+tf1/2) + (tw/2)*math.pow(az1,2)
        Wy2 = b2*tf2*(az2+tf2/2)+(tw/2)*math.pow(az2,2)

        return Wy1+Wy2
    def get_shear_center(self):
        '''
        Returning the shear center
        :return:
        '''
        tf1 = self.plate_th
        tf2 = self.flange_th
        b1 = self.spacing
        b2 = self.flange_width
        h = self.flange_th+self.web_height+self.plate_th
        tw = self.web_th
        hw = self.web_height
        Ax = tf1 * b1 + tf2 * b2 + (h-tf1-tf2) * tw
        # distance to center of gravity in z-direction
        ez = (b2*tf2*tf2/2 + tw*hw*(tf2+hw/2)+tf1*b1*(tf2+hw+tf1/2)) / Ax

        # Shear center:
        # moment of inertia, z-axis
        Iz1 = tf1 * math.pow(b1, 3)
        Iz2 = tf2 * math.pow(b2, 3)
        ht = h - tf1 / 2 - tf2 / 2
        return (Iz1 * ht) / (Iz1 + Iz2) + tf2 / 2 - ez
    def get_moment_of_intertia(self, efficent_se=None):
        '''
        Returning moment of intertia.
        :return:
        '''
        tf1 = self.plate_th
        b1 = self.spacing if efficent_se==None else efficent_se
        h = self.flange_th+self.web_height+self.plate_th
        tw = self.web_th
        hw = self.web_height
        tf2 = self.flange_th
        b2 = self.flange_width

        Ax = tf1 * b1 + tf2 * b2 + (h-tf1-tf2) * tw
        Iyc = (1 / 12) * (b1 * math.pow(tf1, 3) + b2 * math.pow(tf2, 3) + tw * math.pow(hw, 3))
        ez = (tf1 * b1 * (h - tf1 / 2) + hw * tw * (tf2 + hw / 2) + tf2 * b2 * (tf2 / 2)) / Ax
        Iy = Iyc + (tf1 * b1 * math.pow(tf2 + hw + tf1 / 2, 2) + tw * hw * math.pow(tf2 + hw / 2, 2) +
             tf2 * b2 * math.pow(tf2 / 2, 2)) - Ax * math.pow(ez, 2)
        return Iy

    def get_structure_prop(self):
        return self.main_dict

    def get_structure_type(self):
        return self.structure_type

    def get_stiffener_type(self):
        return self.stiffener_type

    def get_shear_area(self):
        '''
        Returning the shear area in [m^2]
        :return:
        '''
        return ((self.flange_th*self.web_th) + (self.web_th*self.plate_th) + (self.web_height*self.web_th))

    def set_main_properties(self, main_dict):
        '''
        Resettting all properties
        :param input_dictionary:
        :return:
        '''

        self.main_dict = main_dict
        self.plate_th = main_dict['plate_thk'][0]
        self.web_height = main_dict['stf_web_height'][0]
        self.web_th = main_dict['stf_web_thk'][0]
        self.flange_width = main_dict['stf_flange_width'][0]
        self.flange_th = main_dict['stf_flange_thk'][0]
        self.mat_yield = main_dict['mat_yield'][0]
        self.mat_factor = main_dict['mat_factor'][0]
        self.span = main_dict['span'][0]
        self.spacing = main_dict['spacing'][0]
        self.structure_type = main_dict['structure_type'][0]
        self.sigma_y1=main_dict['sigma_y1'][0]
        self.sigma_y2=main_dict['sigma_y2'][0]
        self.sigma_x=main_dict['sigma_x'][0]
        self.tauxy=main_dict['tau_xy'][0]
        self.plate_kpp = main_dict['plate_kpp'][0]
        self.stf_kps = main_dict['stf_kps'][0]
        self.km1 = main_dict['stf_km1'][0]
        self.km2 = main_dict['stf_km2'][0]
        self.km3 = main_dict['stf_km3'][0]
        self.stiffener_type=main_dict['stf_type'][0]
        try:
            self.girder_lg=main_dict['girder_lg'][0]
        except KeyError:
            self.girder_lg = 10
        try:
            self.pressure_side = main_dict['press_side'][0]
        except KeyError:
            self.pressure_side = 'p'
        self._zstar_optimization = main_dict['zstar_optimization'][0]
        self._puls_method = main_dict['puls buckling method'][0]
        self._puls_boundary = main_dict['puls boundary'][0]
        self._puls_stf_end  = main_dict['puls stiffener end'][0]
        self._puls_sp_or_up = main_dict['puls sp or up'][0]
        self._puls_up_boundary = main_dict['puls up boundary'][0]

    def set_stresses(self,sigy1,sigy2,sigx,tauxy):
        '''
        Setting the global stresses.
        :param sigy1:
        :param sigy2:
        :param sigx:
        :param tauxy:
        :return:
        '''
        self.main_dict['sigma_y1'][0]= sigy1
        self.sigma_y1 = sigy1

        self.main_dict['sigma_y2'][0]= sigy2
        self.sigma_y2  = sigy2

        self.main_dict['sigma_x'][0]= sigx
        self.sigma_x = sigx

        self.main_dict['tau_xy'][0]= tauxy
        self.tauxy  = tauxy

    def get_plate_thk(self):
        '''
        Return the plate thickness
        :return:
        '''
        return self.plate_th

    def get_cross_section_area(self, efficient_se = None):
        '''
        Returns the cross section area.
        :return:
        '''
        tf1 = self.plate_th
        tf2 = self.flange_th
        b1 = self.spacing if efficient_se==None else efficient_se
        b2 = self.flange_width
        h = self.flange_th+self.web_height+self.plate_th
        tw = self.web_th
        return tf1 * b1 + tf2 * b2 + (h-tf1-tf2) * tw

    def get_cross_section_centroid_with_effective_plate(self, se):
        '''
        Returns cross section centroid
        :return:
        '''
        # checked with example
        tf1 = self.plate_th
        tf2 = self.flange_th
        b1 = se
        b2 = self.flange_width
        h = self.flange_th+self.web_height+self.plate_th
        tw = self.web_th
        hw = self.web_height
        Ax = tf1 * b1 + tf2 * b2 + hw * tw

        return (tf1 * b1 * tf1/2 + hw * tw * (tf1 + hw / 2) + tf2 * b2 * (tf1+hw+tf2/2)) / Ax

    def get_weight(self):
        '''
        Return the weight.
        :return:
        '''
        return 7850*self.span*(self.spacing*self.plate_th+self.web_height*self.web_th+self.flange_width*self.flange_th)

    def get_weight_width_lg(self):
        '''
        Return the weight including Lg
        :return:
        '''
        pl_area = self.girder_lg*self.plate_th
        stf_area = (self.web_height*self.web_th+self.flange_width*self.flange_th)*(self.girder_lg//self.spacing)
        return (pl_area+stf_area)*7850*self.span

    def set_span(self,span):
        '''
        Setting the span. Used when moving a point.
        :return: 
        '''
        self.span = span
        self.main_dict['span'][0] = span

    def get_puls_input(self, run_type: str = 'SP'):
        if self.stiffener_type == 'FB':
            stf_type = 'F'
        else:
            stf_type = self.stiffener_type
        if self._puls_sp_or_up == 'SP':
            return_dict = {'Identification': None, 'Length of panel': self.span*1000, 'Stiffener spacing': self.spacing*1000,
                            'Plate thickness': self.plate_th*1000,
                          'Number of primary stiffeners': 10,
                           'Stiffener type (L,T,F)': stf_type,
                            'Stiffener boundary': self._puls_stf_end,
                          'Stiff. Height': self.web_height*1000, 'Web thick.': self.web_th*1000,
                           'Flange width': self.flange_width*1000,
                            'Flange thick.': self.flange_th*1000, 'Tilt angle': 0,
                          'Number of sec. stiffeners': 0, 'Modulus of elasticity': 2.1e11/1e6, "Poisson's ratio": 0.3,
                          'Yield stress plate': self.mat_yield/1e6, 'Yield stress stiffener': self.mat_yield/1e6,
                            'Axial stress': 0 if self._puls_boundary == 'GT' else self.sigma_x,
                           'Trans. stress 1': 0 if self._puls_boundary == 'GL' else self.sigma_y1,
                          'Trans. stress 2': 0 if self._puls_boundary == 'GL' else self.sigma_y2,
                           'Shear stress': self.tauxy,
                            'Pressure (fixed)': None, 'In-plane support': self._puls_boundary,
                           'sp or up': self._puls_sp_or_up}
        else:
            boundary = self._puls_up_boundary
            blist = list()
            if len(boundary) != 4:
                blist = ['SS', 'SS', 'SS', 'SS']
            else:
                for letter in boundary:
                    if letter.upper() == 'S':
                        blist.append('SS')
                    elif letter.upper() == 'C':
                        blist.append('CL')
                    else:
                        blist.append('SS')

            return_dict = {'Identification': None, 'Length of plate': self.span*1000, 'Width of c': self.spacing*1000,
                           'Plate thickness': self.plate_th*1000,
                         'Modulus of elasticity': 2.1e11/1e6, "Poisson's ratio": 0.3,
                          'Yield stress plate': self.mat_yield/1e6,
                         'Axial stress 1': 0 if self._puls_boundary == 'GT' else self.sigma_x,
                           'Axial stress 2': 0 if self._puls_boundary == 'GT' else self.sigma_x,
                           'Trans. stress 1': 0 if self._puls_boundary == 'GL' else self.sigma_y1,
                         'Trans. stress 2': 0 if self._puls_boundary == 'GL' else self.sigma_y2,
                           'Shear stress': self.tauxy, 'Pressure (fixed)': None, 'In-plane support': self._puls_boundary,
                         'Rot left': blist[0], 'Rot right': blist[1], 'Rot upper': blist[2], 'Rot lower': blist[3],
                           'sp or up': self._puls_sp_or_up}
        return return_dict

    def get_buckling_ml_input(self, design_lat_press: float = 0, sp_or_up: str = 'SP', alone = True, csr = False):
        '''
        Classes in data from ML

        {'negative utilisation': 1, 'non-zero': 2, 'Division by zero': 3, 'Overflow': 4, 'aspect ratio': 5,
        'global slenderness': 6, 'pressure': 7, 'web-flange-ratio': 8,  'below 0.87': 9,
                  'between 0.87 and 1': 10, 'above 1': 11}
        '''
        stf_type = {'T-bar': 1,'T': 1,  'L-bulb': 2, 'Angle': 3, 'Flatbar': 4, 'FB': 4, 'L': 3}
        stf_end = {'Cont': 1, 'C':1 , 'Sniped': 2, 'S': 2}
        field_type = {'Integrated': 1,'Int': 1, 'Girder - long': 2,'GL': 2, 'Girder - trans': 3,  'GT': 3}
        up_boundary = {'SS': 1, 'CL': 2}

        if self._puls_sp_or_up == 'SP':
            if csr == False:
                this_field =  [self.span * 1000, self.spacing * 1000, self.plate_th * 1000, self.web_height * 1000,
                               self.web_th * 1000, self.flange_width * 1000, self.flange_th * 1000, self.mat_yield / 1e6,
                               self.mat_yield / 1e6,  self.sigma_x, self.sigma_y1, self.sigma_y2, self.tauxy,
                               design_lat_press/1000, stf_type[self.stiffener_type], stf_end[self._puls_stf_end]]
            else:
                this_field =  [self.span * 1000, self.spacing * 1000, self.plate_th * 1000, self.web_height * 1000,
                               self.web_th * 1000, self.flange_width * 1000, self.flange_th * 1000, self.mat_yield / 1e6,
                               self.mat_yield / 1e6,  self.sigma_x, self.sigma_y1, self.sigma_y2, self.tauxy,
                               design_lat_press/1000, stf_type[self.stiffener_type], stf_end[self._puls_stf_end],
                               field_type[self._puls_boundary]]
        else:
            ss_cl_list = list()
            for letter_i in self._puls_up_boundary:
                if letter_i == 'S':
                    ss_cl_list.append(up_boundary['SS'])
                else:
                    ss_cl_list.append(up_boundary['CL'])
            b1, b2, b3, b4 = ss_cl_list
            if csr == False:
                this_field =  [self.span * 1000, self.spacing * 1000, self.plate_th * 1000, self.mat_yield / 1e6,
                               self.sigma_x, self.sigma_y1, self.sigma_y2, self.tauxy, design_lat_press/1000,
                               b1, b2, b3, b4]
            else:
                this_field =  [self.span * 1000, self.spacing * 1000, self.plate_th * 1000, self.mat_yield / 1e6,
                               self.sigma_x, self.sigma_y1, self.sigma_y2, self.tauxy, design_lat_press/1000,
                               field_type[self._puls_boundary], b1, b2, b3, b4]
        if alone:
            return [this_field,]
        else:
            return this_field

class CalcScantlings(Structure):
    '''
    This Class does the calculations for the plate fields. 
    Input is a structure object, same as for the structure class.
    The class inherits from Structure class.
    '''
    def __init__(self, main_dict, lat_press = True, category = 'secondary'):
        super(CalcScantlings,self).__init__(main_dict=main_dict)
        self.lat_press = lat_press
        self.category = category
        self._need_recalc = True

    @property
    def need_recalc(self):
        return self._need_recalc

    @need_recalc.setter
    def need_recalc(self, val):
        self._need_recalc = val

    def get_results_for_report(self,lat_press=0):
        '''
        Returns a string for the report.
        :return:
        '''
        buc = [round(res,1) for res in self.calculate_buckling_all(design_lat_press=lat_press)]

        return 'Minimum section modulus:'\
               +str(int(self.get_dnv_min_section_modulus(design_pressure_kpa=lat_press)*1000**3))\
               +'mm^3 '+' Minium plate thickness: '\
               +str(round(self.get_dnv_min_thickness(design_pressure_kpa=lat_press),1))+\
               ' Buckling results: eq7_19: '+str(buc[0])+' eq7_50: '+str(buc[1])+ ' eq7_51: '\
               +str(buc[2])+ ' eq7_52: '+str(buc[3])+ ' eq7_53: '+str(buc[4])

    def calculate_slamming_plate(self, slamming_pressure, red_fac = 1):
        ''' Slamming pressure input is Pa '''
        ka1 = 1.1
        ka2 = min(max(0.4, self.spacing / self.span), 1)

        ka = math.pow(ka1 - 0.25*ka2,2)
        sigmaf = self.mat_yield/1e6  # MPa

        psl = red_fac * slamming_pressure/1000  # kPa
        Cd = 1.5

        return 0.0158*ka*self.spacing*1000*math.sqrt(psl/(Cd*sigmaf))

    def calculate_slamming_stiffener(self, slamming_pressure, angle = 90, red_fac = 1):
        tk = 0
        psl = slamming_pressure / 1000  # kPa
        Pst = psl * red_fac  # Currently DNV does not use psl/2 for slamming.
        sigmaf = self.mat_yield / 1e6  # MPa
        hw, twa, tp, tf, bf, s = [(val - tk) * 1000 for val in [self.web_height, self.web_th, self.plate_th,
                                                                self.flange_th, self.flange_width, self.spacing]]
        ns = 2
        tau_eH = sigmaf/math.sqrt(3)
        h_stf = (self.web_height+self.flange_th)*1000
        f_shr = 0.7
        lbdg = self.span
        lshr = self.span - self.spacing/4000
        dshr = h_stf + tp if 75 <= angle <= 90 else (h_stf + tp)*math.sin(math.radians(angle))
        tw = (f_shr*Pst*s*lshr)/(dshr*tau_eH)

        if self.web_th*1000 < tw:
            return {'tw_req': tw, 'Zp_req':None}
        fpl = 8* (1+(ns/2))
        Zp_req = (1.2*Pst*s*math.pow(lbdg,2)/(fpl*sigmaf)) + \
                  (ns*(1-math.sqrt(1-math.pow(tw/twa,2)))*hw*tw*(hw+tp))/8000

        return {'tw_req': tw, 'Zp_req':Zp_req}

    def check_all_slamming(self, slamming_pressure, stf_red_fact = 1, pl_red_fact = 1):
        ''' A summary check of slamming '''

        pl_chk = self.calculate_slamming_plate(slamming_pressure, red_fac= pl_red_fact)
        if self.plate_th*1000 < pl_chk:
            chk1 = pl_chk / self.plate_th*1000
            return False, chk1

        stf_res = self.calculate_slamming_stiffener(slamming_pressure, red_fac = stf_red_fact)
        #print('Slamming checked')
        if self.web_th*1000 < stf_res['tw_req']:
            chk2 = stf_res['tw_req'] / self.web_th*1000
            return False, chk2

        if stf_res['Zp_req'] is not None:
            eff_pl_sec_mod = self.get_net_effective_plastic_section_modulus()
            if eff_pl_sec_mod < stf_res['Zp_req']:
                chk3 = stf_res['Zp_req']/eff_pl_sec_mod
                return False, chk3

        return True, None

    def get_net_effective_plastic_section_modulus(self, angle = 90):
        ''' Calculated according to Rules for classification: Ships — DNVGL-RU-SHIP Pt.3 Ch.3. Edition July 2017,
            page 83 '''
        tk = 0
        angle_rad = math.radians(angle)
        hw, tw, tp, tf, bf = [(val - tk) * 1000 for val in [self.web_height, self.web_th, self.plate_th, self.flange_th,
                                                            self.flange_width]]
        h_stf = (self.web_height+self.flange_th)*1000
        de_gr = 0
        tw_gr = self.web_th*1000
        hf_ctr = h_stf-0.5*tf if self.get_stiffener_type() not in ['L','L-bulb'] else h_stf - de_gr - 0.5*tf
        bf_ctr = 0 if self.get_stiffener_type() == 'T' else 0.5*(tf - tw_gr)
        beta = 0.5
        gamma = (1 + math.sqrt(3+12*beta))/4

        Af = 0 if self.get_stiffener_type() == 'FB' else bf*tf

        if 75 <= angle <= 90:
            zpl = (hw*tw*(hw+tp)/2000) + ( (2*gamma-1) * Af * ((hf_ctr + tp/2)) / 1000)
        elif angle < 75:
            zpl = (hw*tw*(hw+tp)/2000)+\
                  ( (2*gamma-1) * Af * ((hf_ctr + tp/2) * math.sin(angle_rad) - bf_ctr*math.cos(angle_rad)) / 1000)

        return zpl

    def get_dnv_min_section_modulus(self, design_pressure_kpa, printit = False):
        ''' Section modulus according to DNV rules '''

        design_pressure = design_pressure_kpa
        fy = self.mat_yield / 1e6
        fyd = fy/self.mat_factor

        sigma_y = self.sigma_y2 + (self.sigma_y1-self.sigma_y2)\
                                       *(min(0.25*self.span,0.5*self.spacing)/self.span)

        sigma_jd = math.sqrt(math.pow(self.sigma_x,2)+math.pow(sigma_y,2)-
                             self.sigma_x*sigma_y+3*math.pow(self.tauxy,2))

        sigma_pd2 = fyd-sigma_jd  # design_bending_stress_mpa

        kps = self.stf_kps  # 1 is clamped, 0.9 is simply supported.
        km_sides = min(self.km1,self.km3)  # see table 3 in DNVGL-OS-C101 (page 62)
        km_middle = self.km2  # see table 3 in DNVGL-OS-C101 (page 62)

        Zs = ((math.pow(self.span, 2) * self.spacing * design_pressure) /
              (min(km_middle, km_sides) * (sigma_pd2) * kps)) * math.pow(10, 6)
        if printit:
            print('Sigma y1', self.sigma_y1, 'Sigma y2', self.sigma_y2, 'Sigma x', self.sigma_x, 'Pressure', design_pressure)
        return max(math.pow(15, 3) / math.pow(1000, 3), Zs / math.pow(1000, 3))

    def get_dnv_min_thickness(self, design_pressure_kpa):
        '''
        Return minimum thickness in mm
        :param design_pressure_kpa:
        :return:
        '''

        design_pressure = design_pressure_kpa
        #print(self.sigma_x)
        sigma_y = self.sigma_y2 + (self.sigma_y1-self.sigma_y2)\
                                       *(min(0.25*self.span,0.5*self.spacing)/self.span)
        sigma_jd = math.sqrt(math.pow(self.sigma_x,2)+math.pow(sigma_y,2)-
                             self.sigma_x*sigma_y+3*math.pow(self.tauxy,2))
        fy = self.mat_yield / 1000000
        fyd = fy/self.mat_factor
        sigma_pd1 = min(1.3*(fyd-sigma_jd), fyd)
        sigma_pd1 = abs(sigma_pd1)
        #print(fyd, sigma_jd, fyd)
        if self.category == 'secondary':
            t0 = 5
        else:
            t0 = 7

        t_min = (14.3 * t0) / math.sqrt(fyd)

        ka = math.pow(1.1 - 0.25  * self.spacing/self.span, 2)

        if ka > 1:
            ka =1
        elif ka<0.72:
            ka = 0.72

        assert sigma_pd1 > 0, 'sigma_pd1 must be negative | current value is: ' + str(sigma_pd1)
        t_min_bend = (15.8 * ka * self.spacing * math.sqrt(design_pressure)) / \
                     math.sqrt(sigma_pd1 *self.plate_kpp)

        if self.lat_press:
            return max(t_min, t_min_bend)
        else:
            return t_min

    def get_minimum_shear_area(self, pressure):
        '''
        Calculating minimum section area according to ch 6.4.4.

        Return [m^2]
        :return:
        '''
        #print('SIGMA_X ', self.sigma_x)
        l = self.span
        s = self.spacing
        fy = self.mat_yield

        fyd = (fy/self.mat_factor)/1e6 #yield strength
        sigxd = self.sigma_x #design membrane stresses, x-dir

        taupds = 0.577*math.sqrt(math.pow(fyd, 2) - math.pow(sigxd, 2))

        As = ((l*s*pressure)/(2*taupds)) * math.pow(10,3)

        return As/math.pow(1000,2)

    def is_acceptable_sec_mod(self, section_module, pressure):
        '''
        Checking if the result is accepable.
        :param section_module:
        :param pressure:
        :return:
        '''

        return min(section_module) >= self.get_dnv_min_section_modulus(pressure)

    def is_acceptable_shear_area(self, shear_area, pressure):
        '''
        Returning if the shear area is ok.
        :param shear_area:
        :param pressure:
        :return:
        '''

        return shear_area >= self.get_minimum_shear_area(pressure)

    def get_plate_efficent_b(self,design_lat_press=0,axial_stress=50,
                                 trans_stress_small=100,trans_stress_large=100):
        '''
        Simple buckling calculations according to DNV-RP-C201
        :return:
        '''

        #7.2 Forces in the idealised stiffened plate

        s = self.spacing #ok
        t = self.plate_th #ok
        l = self.span #ok

        E = 2.1e11 #ok

        pSd = design_lat_press*1000
        sigy1Sd =trans_stress_large*1e6
        sigy2Sd =trans_stress_small*1e6
        sigxSd = axial_stress*1e6

        fy = self.mat_yield #ok

        #7.3 Effective plate width
        alphap=0.525*(s/t)*math.sqrt(fy/E) # reduced plate slenderness, checked not calculated with ex
        alphac = 1.1*(s/t)*math.sqrt(fy/E) # checked not calculated with example
        mu6_9 = 0.21*(alphac-0.2)

        if alphac<=0.2: kappa = 1 # eq6.7, all kappa checked not calculated with example
        elif 0.2<alphac<2: kappa = (1/(2*math.pow(alphac,2)))*(1+mu6_9+math.pow(alphac,2)
                                                               -math.sqrt(math.pow(1+mu6_9+math.pow(alphac,2),2)
                                                                          -4*math.pow(alphac,2))) # ok
        else: kappa=(1/(2*math.pow(alphac,2)))+0.07 # ok

        ha = 0.05*(s/t)-0.75
        assert ha>= 0,'ha must be larger than 0'
        kp = 1 if pSd<=2*((t/s)**2)*fy else 1-ha*((pSd/fy)-2*(t/s)**2)

        sigyR=( (1.3*t/l)*math.sqrt(E/fy)+kappa*(1-(1.3*t/l)*math.sqrt(E/fy)))*fy*kp # checked not calculated with example
        l1 = min(0.25*l,0.5*s)

        sig_min, sig_max = min(sigy1Sd,sigy2Sd),max(sigy1Sd,sigy2Sd) # self-made
        sigySd = sig_min+(sig_max-sig_min)*(1-l1/l) # see 6.8, page 15

        ci = 1-s/(120*t) if (s/t)<=120 else 0 # checked not calculated with example

        Cxs = (alphap-0.22)/math.pow(alphap,2) if alphap > 0.673 else 1 # reduction factor longitudinal
        # eq7.16, reduction factor transverse, compression (positive) else tension

        Cys = math.sqrt(1-math.pow(sigySd/sigyR,2) + ci*((sigxSd*sigySd)/(Cxs*fy*sigyR))) if sigySd >= 0 \
            else min(0.5*(math.sqrt(4-3*math.pow(sigySd/fy,2))+sigySd/fy),1) #ok, checked

        #7.7.3 Resistance parameters for stiffeners
        return s * Cxs * Cys # 7.3, eq7.13, che

    def calculate_buckling_all(self,design_lat_press=0.0, checked_side = 'p'):
        '''
        Simple buckling calculations according to DNV-RP-C201
        :return:
        '''
        #7.2 Forces in the idealised stiffened plate
        As = self.web_height*self.web_th+self.flange_width*self.flange_th #checked
        s = self.spacing #ok
        t = self.plate_th #ok
        l = self.span #ok
        tf = self.flange_th
        tw = self.web_th
        hw = self.web_height
        bf = self.flange_width
        fy = self.mat_yield  # ok
        stf_type = self.get_stiffener_type()
        zstar = self._zstar_optimization  # simplification as per 7.7.1 Continuous stiffeners

        E = 2.1e11 #ok
        Lg = 10 #girder length, ok
        mc = 13.3  # assume continous stiffeners

        pSd = design_lat_press*1000
        tauSd = self.tauxy*1e6
        sigy1Sd =self.sigma_y1*1e6
        sigy2Sd =self.sigma_y2*1e6
        sigxSd = self.sigma_x*1e6


        #7.3 Effective plate width
        alphap=0.525*(s/t)*math.sqrt(fy/E) # reduced plate slenderness, checked not calculated with ex
        alphac = 1.1*(s/t)*math.sqrt(fy/E) # eq 6.8 checked not calculated with example
        mu6_9 = 0.21*(alphac-0.2)

        #kappa chapter 6.3
        if alphac<=0.2: kappa = 1 # eq6.7, all kappa checked not calculated with example
        elif 0.2<alphac<2: kappa = (1/(2*math.pow(alphac,2))) * (1+mu6_9+math.pow(alphac,2)
                                                                 - math.sqrt(math.pow(1+mu6_9+math.pow(alphac,2),2)
                                                                             -4*math.pow(alphac,2))) # ok
        else: kappa=(1/(2*math.pow(alphac,2)))+0.07 # ok
        #end kappa

        ha = 0.05*(s/t)-0.75 #eq 6.11 - checked, ok

        #assert ha>= 0,'ha must be larger than 0'
        if ha < 0:
            return [0, float('inf'), 0, 0, 0, 0]

        kp = 1 if pSd<=2*math.pow(t/s,2)*fy else max(1-ha*((pSd/fy)-2*math.pow(t/s,2)),0) #eq 6.10, checked

        sigyR=( (1.3*t/l)*math.sqrt(E/fy)+kappa*(1-(1.3*t/l)*math.sqrt(E/fy)))*fy*kp # eq 6.6 checked

        sigyRd = sigyR / self.mat_factor #eq 6.5 checked, ok


        # plate resistance check
        ksp = math.sqrt(1-3*math.pow(tauSd/(fy/1),2)) #eq7.20 ch7.4, checked ok

        l1 = min(0.25*l,0.5*s)
        sig_min, sig_max = min(sigy1Sd,sigy2Sd),max(sigy1Sd,sigy2Sd) # self-made
        sigySd = sig_min+(sig_max-sig_min)*(1-l1/l) # see 6.8, page 15

        if not sigySd <= sigyRd:
            return [float('inf'),0,0,0,0,0]

        try:
            psi = sigy2Sd/sigy1Sd # eq. 7.11 checked, if input is 0, the psi is set to 1
        except ZeroDivisionError:
            psi = 1

        Is = self.get_moment_of_intertia()  # moment of intertia full plate width
        Ip = math.pow(t,3)*s/10.9 # checked not calculated with example

        kc = 2*(1+math.sqrt(1+(10.9*Is)/(math.pow(t,3)*s))) # checked not calculated with example
        kg = 5.34+4*math.pow((l/Lg),2) if l<=Lg else 5.34*math.pow(l/Lg,2)+4 # eq 7.5 checked not calculated with example
        kl = 5.34+4*math.pow((s/l),2) if l>=s else 5.34*math.pow(s/l,2)+4 # eq7.7 checked not calculated with example

        taucrg = kg*0.904*E*math.pow(t/l,2) # 7.2 critical shear stress, checked not calculated with example
        taucrl = kl*0.904*E*math.pow(t/s,2) # 7.2 critical chear stress, checked not calculated with example
        tautf = (tauSd - taucrg) if  tauSd>taucrl/self.mat_factor else 0 # checked not calculated with example

        #7.6 Resistance of stiffened panels to shear stresses (page 20)
        taucrs = (36*E/(s*t*math.pow(l,2)))*((Ip*math.pow(Is,3))**0.25) # checked not calculated with example
        tauRd = min(fy/(math.sqrt(3)*self.mat_factor), taucrl/self.mat_factor,taucrs/self.mat_factor)# checked not calculated with example

        ci = 1-s/(120*t) if (s/t)<=120 else 0 # checked ok
        Cxs = (alphap-0.22)/math.pow(alphap,2) if alphap>0.673 else 1 # reduction factor longitudinal, ok

        # eq7.16, reduction factor transverse, compression (positive) else tension

        Cys = math.sqrt(1-math.pow(sigySd/sigyR,2)+ci*((sigxSd*sigySd)/(Cxs*fy*sigyR))) if sigySd >= 0 \
            else min(0.5*(math.sqrt(4-3*math.pow(sigySd/fy,2))+sigySd/fy),1) #eq 7.16, ok, checked
        #7.7.3 Resistance parameters for stiffeners

        se = s * Cxs * Cys # 7.3, eq7.13, checked
        zp = self.get_cross_section_centroid_with_effective_plate(se) - t / 2  # ch7.5.1 page 19
        zt = (t / 2 + hw + tf) - zp  # ch 7.5.1 page 19

        Ie = self.get_moment_of_intertia(efficent_se=se) #ch7.5.1 effective moment of inertia.
        Wep = Ie/zp #as def in eq7.71
        Wes = Ie/zt #as def in eq7.71

        C0 = (Wes * fy * mc) / (kc * E * math.pow(t, 2) * s)  # 7.2 checked not calculated with example
        p0 = (0.6+0.4*psi)*C0*sigy1Sd if psi>-1.5 else 0 # 7.2 checked not calculated with example

        qSd = (pSd + p0) * s  # checked not calculated with example

        Ae = As+se*t #ch7.7.3 checked, ok

        W = min(Wes,Wep) #eq7.75 text, checked
        pf = (12*W/(math.pow(l,2)*s))*(fy/self.mat_factor) #checked, ok

        lk = l*(1-0.5*abs(pSd/pf)) #eq7.74, buckling length, checked

        ie = math.sqrt(Ie/Ae) #ch 7.5.1. checked
        fE = math.pow(math.pi,2)*E*math.pow(ie/lk,2) #e7.24, checked

        sigjSD = math.sqrt(math.pow(sigxSd,2)+math.pow(sigySd,2)-sigxSd*sigySd+3*math.pow(tauSd,2)) # eq 7.38, ok
        fEpx = 3.62*E*math.pow(t/s,2) # eq 7.42, checked, ok
        fEpy = 0.9*E*math.pow(t/s,2) # eq 7.43, checked, ok
        fEpt = 5.0*E*math.pow(t/s,2) # eq 7.44, checked, ok
        c = 2-(s/l) # eq 7.41, checked, ok
        try:
            alphae = math.sqrt( (fy/sigjSD) * math.pow(math.pow(abs(sigxSd)/fEpx, c)+
                                                       math.pow(abs(sigySd)/fEpy, c)+
                                                       math.pow(abs(tauSd)/fEpt, c), 1/c)) # eq 7.40, checed, ok.
        except OverflowError:
            import tkinter as tk
            tk.messagebox.showerror('Error', 'There is an issue with your input. \n'
                                             'Maybe a dimension is nor correct w.r.t.\n'
                                             'm and mm. Check it!\n\n'
                                             'A plate resistance error will be shown\n'
                                             'for buckling. This is not correct but is\n'
                                             'due to the input error.')
            return [float('inf'),0,0,0,0,0]
        fep = fy / math.sqrt(1+math.pow(alphae,4)) # eq 7.39, checked, ok.
        eta = min(sigjSD/fep, 1) # eq. 7.37, chekced

        C = (hw / s) * math.pow(t / tw, 3) * math.sqrt((1 - eta)) # e 7.36, checked ok

        beta = (3*C+0.2)/(C+0.2) # eq 7.35, checked, ok

        Af = self.flange_width*self.flange_th #flange area, ok
        Aw = self.web_height*self.web_th #web area, ok

        ef = 0 if stf_type in ['FB','T'] else self.flange_width/2-self.web_th/2
        #Ipo = (Aw*(ef-0.5*tf)**2/3+Af*ef**2)*10e-4 #polar moment of interia in cm^4
        #It = (((ef-0.5*tf)*tw**3)/3e4)*(1-0.63*(tw/(ef-0.5*tf)))+( (bf*tf)/3e4*(1-0.63*(tf/bf)))/(100**4) #torsonal moment of interia cm^4


        Iz = (1/12)*Af*math.pow(bf,2)+math.pow(ef,2)*(Af/(1+(Af/Aw))) #moment of inertia about z-axis, checked

        G = E/(2*(1+0.3)) #rules for ships Pt.8 Ch.1, page 334
        lT = self.span # Calculated further down
        #print('Aw ',Aw,'Af ', Af,'tf ', tf,'tw ', tw,'G ', G,'E ', E,'Iz ', Iz,'lt ', lT)

        def get_some_data(lT):
            if stf_type in ['T', 'L', 'L-bulb']:
                fET = beta*(((Aw + Af * math.pow(tf/tw,2)) / (Aw + 3*Af)) * G*math.pow(tw/hw,2))+\
                      (math.pow(math.pi, 2) * E * Iz) / ((Aw/3 + Af)*math.pow(lT,2)) \
                    if bf != 0 \
                    else (beta+2*math.pow(hw/lT,2))*G*math.pow(tw/hw,2) # eq7.32 checked, no example
            else:
                fET = (beta + 2*math.pow(hw/lT,2))*G*math.pow(tw/hw,2) # eq7.34 checked, no example

            alphaT = math.sqrt(fy/fET) #eq7.30. checked

            mu7_29 = 0.35 * (alphaT - 0.6) # eq 7.29. checked

            fr = fy if alphaT<=0.6 else ((1+mu7_29+math.pow(alphaT,2)-math.sqrt( math.pow(1+mu7_29+math.pow(alphaT,2),2)-
                                                                                 4*math.pow(alphaT,2))) /
                                         (2*math.pow(alphaT,2))) * fy
            alpha = math.sqrt(fr / fE) #e7.23, checked.

            mu_tors = 0.35*(alphaT-0.6)
            fT = fy if alphaT <= 0.6 else fy * (1+mu_tors+math.pow(alphaT,2)-math.sqrt(math.pow(1+mu_tors+math.pow(alphaT,2),2)-
                                                                                     4*math.pow(alphaT,2)))/\
                                           (2*math.pow(alphaT,2))

            mu_pl = (0.34 + 0.08 * (zp / ie)) * (alpha - 0.2)
            mu_stf = (0.34 + 0.08 * (zt / ie)) * (alpha - 0.2)
            frp = fy
            frs = fy if alphaT <= 0.6 else fT
            fyp,fys = fy,fy
            #fyps = (fyp*se*t+fys*As)/(se*t+As)
            fks = fr if alpha <= 0.2 else frs * (1+mu_stf+math.pow(alpha,2)-math.sqrt(math.pow(1+mu_stf+math.pow(alpha,2),2)-
                                                                                     4*math.pow(alpha,2)))/\
                                          (2*math.pow(alpha,2))
            #fr = fyps
            fkp = fyp if alpha <= 0.2 else frp * (1+mu_pl+math.pow(alpha,2)-math.sqrt(math.pow(1+mu_pl+math.pow(alpha,2),2)-
                                                                                     4*math.pow(alpha,2)))/\
                                           (2*math.pow(alpha,2))

            return fr, fks, fkp

        u = math.pow(tauSd / tauRd, 2)  # eq7.58. checked.
        fr, fks, fkp = get_some_data(lT=lT*0.4)
        Ms1Rd = Wes*(fr/self.mat_factor) #ok, assuming fr calculated with lT=span * 0.4
        NksRd = Ae * (fks / self.mat_factor) #eq7.66, page 22 - fk according to equation 7.26, sec 7.5,
        NkpRd = Ae * (fkp / self.mat_factor)  # checked ok, no ex

        M1Sd = abs((qSd*math.pow(l,2))/12) #ch7.7.1, checked ok

        M2Sd = abs((qSd*l**2)/24) #ch7.7.1, checked ok

        Ne = ((math.pow(math.pi,2))*E*Ae)/(math.pow(lk/ie,2))# eq7.72 , checked ok

        Nrd = Ae * (fy / self.mat_factor) #eq7.65, checked ok

        Nsd = sigxSd * (As + s*t) + tautf * s *t #  Equation 7.1, section 7.2, checked ok


        MstRd = Wes*(fy/self.mat_factor) #eq7.70 checked ok, no ex
        MpRd = Wep*(fy/self.mat_factor) #eq7.71 checked ok, no ex

        fr, fks, fkp = get_some_data(lT = lT * 0.8)
        Ms2Rd = Wes*(fr/self.mat_factor) #eq7.69 checked ok, no ex
        # print('Nksrd', NksRd, 'Nkprd', NkpRd, 'Ae is', Ae, 'fks is', fks, 'fkp is', fkp,
        #       'alphas are', mu_pl, mu_stf, 'lk', lk, 'lt', lT)

        #print('CENTROID ', 'zp', 'zt', self.get_cross_section_centroid_with_effective_plate(se)*1000,zp,zt)

        eq7_19 = sigySd/(ksp*sigyRd) #checked ok
        if self._zstar_optimization:
            zstar_range = np.arange(-zt/2,zp,0.002)
        else:
            zstar_range = [0]
        # Lateral pressure on plate side:
        if checked_side == 'p':
            # print('eq7_50 = ',Nsd ,'/', NksRd,'+' ,M1Sd,'-' , Nsd ,'*', zstar, '/' ,Ms1Rd,'*',1,'-', Nsd ,'/', Ne,'+', u)
            # print('eq7_51 = ',Nsd,' / ',NkpRd,' - 2 * ',Nsd, '/' ,Nrd,' + ',M1Sd,' - ,',Nsd,' * ',zstar,' / ',MpRd,' * ','1 - ',Nsd,' / ',Ne,' + ',u)
            #print('eq7_52 = ',Nsd,'/', NksRd,'-', 2, '*',Nsd,'/', Nrd,'+',M2Sd,'-', Nsd,'*', zstar,'/',MstRd,'*',1, '-',Nsd,'/', Ne,'+', u)
            max_lfs = []
            ufs = []
            for zstar in zstar_range:
                eq7_50 = (Nsd / NksRd) + (M1Sd - Nsd * zstar) / (Ms1Rd * (1 - Nsd / Ne)) + u
                eq7_51 = (Nsd / NkpRd) - 2 * (Nsd / Nrd) + ((M1Sd - Nsd * zstar) / (MpRd * (1 - (Nsd / Ne)))) + u
                eq7_52 = (Nsd / NksRd) - 2 * (Nsd / Nrd) + ((M2Sd + Nsd * zstar) / (MstRd * (1 - (Nsd / Ne)))) + u
                eq7_53 = (Nsd / NkpRd) + (M2Sd + Nsd * zstar) / (MpRd * (1 - Nsd / Ne))
                max_lfs.append(max(eq7_50, eq7_51, eq7_52, eq7_53))
                ufs.append([eq7_19, eq7_50, eq7_51, eq7_52, eq7_53,zstar])
                #print(zstar, eq7_50, eq7_51, eq7_52, eq7_53, 'MAX LF is: ', max(eq7_50, eq7_51, eq7_52, eq7_53))
            min_of_max_ufs_idx = max_lfs.index(min(max_lfs))
            return ufs[min_of_max_ufs_idx]
        # Lateral pressure on stiffener side:
        else:
            max_lfs = []
            ufs = []
            for zstar in zstar_range:
                eq7_54 = (Nsd / NksRd) - 2 * (Nsd / Nrd) + ((M1Sd + Nsd * zstar) / (MstRd * (1 - (Nsd / Ne)))) + u
                eq7_55 = (Nsd / NkpRd) + ((M1Sd + Nsd * zstar) / (MpRd * (1 - (Nsd / Ne)))) + u
                eq7_56 = (Nsd / NksRd) + ((M2Sd - Nsd * zstar) / (Ms2Rd * (1 - (Nsd / Ne)))) + u
                eq7_57 = (Nsd / NkpRd) - 2 * (Nsd / Nrd) + ((M2Sd - Nsd * zstar) / (MpRd * (1 - (Nsd / Ne)))) + u
                max_lfs.append(max(eq7_54, eq7_55, eq7_56, eq7_57))
                ufs.append([eq7_19, eq7_54, eq7_55, eq7_56, eq7_57, zstar])
                #print('eq7_19, eq7_54, eq7_55, eq7_56, eq7_57')
            min_of_max_ufs_idx = max_lfs.index(min(max_lfs))
            return ufs[min_of_max_ufs_idx]


    def calculate_buckling_plate(self,design_lat_press,axial_stress=20,
                                 trans_stress_small=100,trans_stress_large=100,
                                 design_shear_stress = 10):
        '''
        Simple buckling calculations according to DNV-RP-C201
        This method is currently not used.
        :return:
        '''

        #7.2 Forces in the idealised stiffened plate

        s = self.spacing
        t = self.plate_th
        l = self.span
        E = 2.1e11

        pSd = design_lat_press*1000
        tauSd = design_shear_stress*1e6
        sigy2Sd =trans_stress_small*1e6
        fy = self.mat_yield

        #7.3 Effective plate width
        alphac = 1.1*(s/t)*math.sqrt(fy/E)
        gamma = 0.21*(alphac-0.2)

        if alphac<=0.2: kappa = 1
        elif 0.2<alphac<2: kappa = (1/(2*(alphac**2)))*(1-+gamma+alphac**2-math.sqrt((1+gamma+alphac**2)**2-4*alphac**2))
        else: kappa=(1/(2*alphac**2))+0.7

        ha = 0.05*(s/t)-0.75
        assert ha>= 0,'ha must be larger than 0'
        kp = 1 if pSd<=2*((t/s)**2)*fy else 1-ha*((pSd/fy)-2*(t/s)**2)

        sigyR=( (1.3*t/l)*math.sqrt(E/fy)+kappa*(1-(1.3*t/l)*math.sqrt(E/fy)))*fy*kp
        sigyRd = sigyR / self.mat_factor

        # plate resistance check
        ksp = math.sqrt(1-3*(tauSd/(fy/1))**2)
        eq7_19 = ksp*sigyRd/sigy2Sd
        return eq7_19

    def buckling_local_stiffener(self):
        '''
        Local requirements for stiffeners. Chapter 9.11.
        :return:
        '''

        epsilon = math.sqrt(235 / (self.mat_yield / 1e6))

        if self.stiffener_type in ['L', 'L-bulb']:
            c = self.flange_width - self.web_th/2
        elif self.stiffener_type == 'T':
            c = self.flange_width/2 - self.web_th/2
        elif self.stiffener_type == 'FB':
            return self.web_height <= 42 * self.web_th * epsilon, self.web_height/(42 * self.web_th * epsilon)

        # print(self.web_height, self.web_th, self.flange_width ,self.flange_th )
        # print('c:',c, 14 * self.flange_th * epsilon, ' | ',  self.web_height, 42 * self.web_th * epsilon)
        # print(c <= (14  * self.flange_th * epsilon) and self.web_height <= 42 * self.web_th * epsilon)
        # print(c/(14  * self.flange_th * epsilon), self.web_height / (42 * self.web_th * epsilon))
        # print('')

        return c <= (14  * self.flange_th * epsilon) and self.web_height <= 42 * self.web_th * epsilon, \
               max(c/(14  * self.flange_th * epsilon), self.web_height / (42 * self.web_th * epsilon))

    def is_acceptable_pl_thk(self, design_pressure):
        '''
        Checking if the thickness is acceptable.
        :return:
        '''
        return self.get_dnv_min_thickness(design_pressure) <= self.plate_th*1000

    def buckling_stiffened_cylinder(self):
        '''
        Calculate bucling of cylinder

        σx,Sd = #design membrane stress in the longitudinal direction (tension is positive)
        σh,Sd = #design membrane stress in the circumferential direction (tension is positive)
        τSd =   #design shear stress tangential to the shell surface (in sections x = constant and θ = constant)
        NSd = #Design axial force
        MSd = #Design bending moments
        TSd = #Design torsional moment
        QSd = #Design shear force
        pSd = #Design lateral pressure
        θ = 0 #circumferential co-ordinate measured from axis 1
        A = #cross-sectional area of a longitudinal stiffener (exclusive of shell flange)
        Ac cross sectional area of complete cylinder section; including longitudinal stiffeners/internal bulkheads if any
        Af cross sectional area of flange (=btf)
        AR cross-sectional area of a ring frame (exclusive of shell flange)
        AReq required cross sectional area (exclusive of effective plate flange) of ring
        s = 0 #distance between longitudinal stiffeners
        '''
        sigxSd = None#design membrane stress in the longitudinal direction (tension is positive)
        sighSd = None#design membrane stress in the circumferential direction (tension is positive)
        tauSd =  None #design shear stress tangential to the shell surface (in sections x = constant and θ = constant)
        NSd = None#Design axial force
        MSd = None#Design bending moments
        M1Sd = None
        M2Sd = None

        TSd = None#Design torsional moment
        QSd = None#Design shear force
        Q1Sd = None
        Q2Sd = None

        pSd = None#Design lateral pressure
        r = None # Radius of cylinder.
        rr = None # Radius (variable)
        t = None # Thickness of cylinder.
        theta = None  # circumferential co-ordinate measured from axis 1, θ

        A = None # cross-sectional area of a longitudinal stiffener (exclusive of shell flange)
        Ar = None #cross-sectional area of a ring frame (exclusive of shell flange)
        s = None  # distance between longitudinal stiffeners
        l = None # distance between ring frames
        lt = None # torsional buckling length
        leq = None # equivalent lenth

        h = None # Web height
        tw = None # Web thickness
        b = None # Flange width


        v = 0.3 # Poisson

        long_stiff = True
        bhd = False

        checks = dict()
        #--------------------2.2.2 Longitudinal membrane stress-------------------------

        #For a cylindrical shell without longitudinal stiffeners:
        sigaSd = NSd/(2*math.pi*r*t) #(2.2.2)
        sigmSd = (M1Sd/(math.pi*math.pow(r,2)*t))*math.sin(theta) - \
                 (M2Sd/(math.pi*math.pow(r,2)*t))*math.cos(theta) #(2.2.3)

        #For a cylindrical shell with longitudinal stiffeners it is usually permissible to replace the shell thickness by the
        #equivalent thickness for calculation of longitudinal membrane stress only:

        te = t + A/s #(2.2.4)
        sigxSd = sigaSd + sigmSd #(2.2.1)

        # --------------------2.2.3 Shear stress-------------------------

        tauTSd= TSd/(2*math.pi*math.pow(r,2)*t) #(2.2.6)

        tauQsd = (Q1Sd/(math.pi*r*t))*math.sin(theta) - \
                 (Q2Sd/(math.pi*r*t))*math.cos(theta) #(2.2.7)

        tauSd = tauTSd + tauQsd #(2.2.5)

        # --------------------2.2.4 Circumferential membrane stress-------------------------
        #For an unstiffened cylinder the circumferential membrane stress may be taken as:

        sighSd = math.pow(pSd, r)/t #(2.2.8)

        #For a ringstiffened cylinder (without longitudinal stiffeners) the circumferential membrane stress midway
        #between two ring frames may be taken as:

        beta = l/(1.56*math.sqrt(r*t)) #(2.2.11)
        alpha = Ar/(leq*t) #(2.2.12)
        leq = (l/beta) * ( (math.cosh(2*beta) - math.cos(2*beta)) / (math.sinh(2*beta) + math.sin(2*beta))) # (2.2.13)
        zeta = max(0.0, 2*( (math.sinh(beta)*math.cos(beta) + math.cosh(beta)*math.sin(beta)) / (math.sinh(2*beta) + math.sin(2*beta)))) #(2.2.10)
        sighSd = math.pow(pSd, r) / t + (alpha*zeta/(alpha+1)) * ( (math.pow(pSd, r)/t) - v*sigxSd) #(2.2.9)

        # --------------------2.2.5 Circumferential stress in a ring frame-------------------------
        #For ring stiffened shells the circumferential stress in a ring frame at the distance rr (rr is variable, rr = rf at ring
        #flange position and rr = r at shell) from the cylinder axis may be taken as:

        if long_stiff:
            this_alpha = A/lt
        else:
            this_alpha = alpha

        sighRSd = (pSd*r/r - v*sigxSd)*(1/(1+this_alpha))*(r/rr) #(2.2.15)

        # --------------------2.2.6 Stresses in shells at bulkheads and ring stiffeners-------------------------
        #The circumferential membrane stress at a ring frame for a ring stiffened cylinder (without longitudinal
        #stiffeners) may be taken as:
        # TODO provision for bulkeheads

        #2.2.6.2 Circumferential membrane stress
        sighSd = (pSd*r/t-v*sigxSd) * (1/(1+alpha)) + v*sigxSd #(2.2.17)

        #2.2.6.3 Bending stress

        sigxmSd = (math.pow(pSd, r)/t - sighSd)*math.sqrt(3/(1-v)) #(2.2.19) where σh,Sd is given in (2.2.17) or (2.2.18).

        #The circumferential bending stress in the shell at a bulkhead or a ring frame is:

        sighmSd = v*sigxmSd #(2.2.20)

        # --------------------3 Buckling Resistance of Cylindrical Shells-------------------------

        sigjsd = None

        fks = None

        #3.2 Characteristic buckling strength of shells
        fy = self.mat_yield/1.6

        siga0sd = 0 if sigaSd >= 0 else -sigaSd #(3.2.4)
        sigm0sd = 0 if sigmSd >= 0 else -sigmSd #(3.2.5)
        sigh0sd = 0 if sighSd >= 0 else -sighSd #(3.2.6)

        fEa = None
        fEm = None
        fEh = None
        fEtau = None

        lambdas = (fy/sigjsd)* (siga0sd/fEa + sigm0sd/fEm + sigh0sd/fEh + tauSd/fEtau) #(3.2.2)
        fks = fy / math.sqrt(1 + lambdas) #(3.2.1)
        if lambdas<0.5:
            gammaM = 1.15 #(3.1.3)
        elif 0.5 <= lambdas <= 1:
            gammaM = 0.85 + 0.6*lambdas #(3.1.3)
        else:
            gammaM = 1.45 #(3.1.3)
        fksd = fks / gammaM #(3.1.2)
        sigjsd = math.sqrt(math.pow(sigaSd+sigmSd, 2) - math.pow(sigaSd+sigmSd)*sighSd + math.pow(sighSd, 2)+
                           3*math.pow(tauSd, 2)) #(3.2.3)

        # --------------------3.3 Elastic buckling strength of unstiffened curved panels-------------------------
        Zs = (math.pow(s, 2)/(r*t)) * math.sqrt(1 - math.pow(v, 2)) #The curvature parameter Zs (3.3.3)

        #Table 3-1 Buckling coefficient for unstiffened curved panels, mode a) Shell buckling
        psi = {'Axial stress': 4, 'Shear stress': 5.54+4*math.pow(s/l, 2),
               'Circumferential compression': math.pow(1+math.pow(s/l, 2), 2)}                      # ψ
        ksi = {'Axial stress': 0.702*Zs, 'Shear stress': 0.856*math.sqrt(s/l)*math.pow(Zs, 3/4),
               'Circumferential compression': 1.04*(s/l)*math.sqrt(Zs)}                             # ξ
        rho = {'Axial stress': 0.5*math.pow(1+(r/(150*t)), -0.5), 'Shear stress': 0.6,
               'Circumferential compression': 0.6}
        # End table 3-1

        E = 2.1e11/1e6

        C = psi*math.sqrt(1+math.pow(rho*ksi/psi, 2)) #(3.3.2)
        fE = C * ( (math.pow(math.pi, 2)*E) / (12*(1-math.pow(v, 2)) )) * math.pow(t/s, 2) #(3.3.1)

        # --------------------3.4 Elastic buckling strength of unstiffened circular cylinders-------------------------

        #3.4.2 Shell buckling
        fE = C * ( (math.pow(math.pi, 2)*E) / (12*(1-math.pow(v, 2)) )) * math.pow(t/l, 2) #(3.4.1) (3.6.3)
        C = psi * math.sqrt(1 + math.pow(rho * ksi / psi, 2))  # (3.4.2) (3.6.4)

        Zl = (math.pow(l, 2)/(r*t)) * math.sqrt(1 - math.pow(v, 2)) #(3.4.3) (3.6.5)

        long_cylinder = False

        #Buckling coefficients for unstiffened cylindrical shells, mode a) Shell buckling
        psi = {'Axial stress': 1, 'Bending': 1,
               'Torsion and shear force': 5.34,
               'lateral pressure': 4, 'Hydrostatic pressure': 2}                      # ψ

        ksi = {'Axial stress': 0.702*Zl, 'Bending': 0.702*Zl,
               'Torsion and shear force': 0.856* math.pow(Zl, 3/4),'lateral pressure': 1.04*math.sqrt(Zl),
               'Hydrostatic pressure': 1.04*math.sqrt(Zl)} # ξ

        rho = {'Axial stress': 0.5*math.pow(1+(r/(150*t)), -0.5), 'Bending': 0.5*math.pow(1+(r/(300*t)), -0.5),
               'Torsion and shear force': 0.6,
               'lateral pressure': 0.6, 'Hydrostatic pressure': 0.6}
        #End table 3-2

        if l/r > 3.85*math.sqrt(r/t):
            fEtau = 0.25 * E * math.pow(t/r, 3/2) #(3.4.4)
        if l/r > 2.25*math.sqrt(r/t):
            fEh = 0.25*E*math.pow(t/r, 2) #(3.4.5)

        # --------------------3.5 Ring stiffened shells-------------------------
        checks['3.5 Ring stiffened shells'] = dict()
        #Panel ring buckling
        checks['3.5 Ring stiffened shells']['3.5.2 Panel ring buckling'] = dict()
        ##Cross sectional area.

        Areq = None
        checks['3.5 Ring stiffened shells']['3.5.2 Panel ring buckling']['3.5.2.1 Cross sectional area'] = \
            Areq >= (2 / math.pow(Zl, 2) + 0.06) * l * t #(3.5.1)

        ##Moment of inertia
        Ix = None
        Ixh = None
        Ih = None

        Ir = Ix + Ixh + Ih #(3.5.2)

        lef = min((1.56 * math.sqrt(r*t)) / (1+12*(t/r)), l) # (3.5.3), (3.5.4)

        ## Calculation of Ix
        alphaA = A/(s*t) #(3.5.6) A = cross sectional area of a longitudinal stiffener.
        r0 = None # r0 radius of the shell measured to the neutral axis of ring frame with effective shell flange, leo
        Ix = abs(sigxSd)*t*(1+alphaA)*math.pow(r0, 4) / (500*E*l) #(3.5.5)

        ## 3.5.2.5 Calculation of Ixh
        L = None # distance between effective supports of the ring stiffened cylinder
        Ixh = math.pow(tauSd/E, 8/5) * math.pow(r0/L, 1/5) *L*r0*t*l #(3.5.7)

        ##General calculation of Ih for external pressure
        zt = None
        delta0 = 0.005*r #(3.5.13)

        fT = None       #The torsional buckling strength,
        fabricated = True
        fr = fT if fabricated else 0.9*fT

        Ih = ( (abs(pSd)*r*math.pow(r0,2)*l) / (3*E) ) * (1.5 + ( (3*E*zt*delta0) /
                                                                  (math.pow(r0,2)*(fr/2)-abs(sighRSd) ))) #(3.5.8) TODO criteras here

        #The torsional buckling strength, fT, may be taken equal to the yield strength, fy, if the following
        # requirements are satisfied: TODO type of stiffener must be specified

        chk1 = h <= 0.4*tw*math.sqrt(E/fy) #(3.5.9) (3.5.17)
        chk2 = h <= 1.35 * tw * math.sqrt(E / fy) #(3.5.10) (3.5.18)
        chk3 = b >= (7*h) / math.sqrt(10+(E/fy)*(h/r)) #(3.5.11) (3.5.19)

        if all(chk1, chk2, chk3):
            fT = fy
            fr = fy
        else:
            fT = None # TODO chapter 3.9
            fr = None  # TODO chapter 3.9, fT

        w = delta0*math.cos(2*theta) #(3.5.12)

        ## Special calculation of Ih for external pressure
        fk = None
        rf = None #radius of the shell measured to the ring flange,
        chk_psd = abs(pSd) <= 0.75 * (fk/gammaM) * ( (t*rf*(1+(Ar/(leq*t)))) / (math.pow(r,2)*(1-v/2))  )

        checks['3.5 Ring stiffened shells']['3.5.2 Panel ring buckling']['Special calculation of Ih for external pressure'] = chk_psd

        # fK is the characteristic buckling strength found from:

        ZL = (math.pow(L, 2)/(r*t)) * math.sqrt(1-math.pow(v,2)) # (3.5.22)
        alphaB = (12*(1-math.pow(v,2))*Ih) / (l*math.pow(t,3)) # (3.5.23)
        alpha = Ar/(leq*t) # (3.5.24)
        C1 = ( 2*(1+alphaB)/(1+alpha) ) * ( math.sqrt(1+(0.27*ZL)/math.sqrt(1+alphaB)) - alphaB/(1+alphaB) ) #(3.5.21)
        C2 = 2 * math.sqrt(1 + 0.27 * ZL)
        ih = math.sqrt(Ih / (Ar + leq*t)) # (3.5.27)
        mu = (zt*delta0/math.pow(ih, 2)) * (rf/r) * (l/leq) * (1-C2/C1) * (1/(1-(v/2))) # μ (3.5.25)

        alpha_overline = math.sqrt(fr/fE) #(3.5.16)

        fK = ( (1+mu+math.pow(alpha_overline,2)-math.sqrt(math.pow(1+mu+math.pow(alpha_overline,2),2)-
                                                          4*math.pow(alpha_overline,2))) /
               (2*math.pow(alpha_overline, 2)) ) * fr   #(3.5.15)

        #The characteristic material strength, fr, may be taken equal to the yield strength, fy, if the following
        # requirements are satisfied: TODO type of stiffener must be specified

        fE = C1 * ( (math.pow(math.pi,2)*E)/(12*(1-math.pow(v,2))) ) * math.pow(t/L, 2) # (3.5.20)

        # --------------------3.6 Longitudinally stiffened shells-------------------------
        # 3.6.3 Panel stiffener buckling
        checks['3.6 Longitudinally stiffened shells'] = dict()
        checks['3.6 Longitudinally stiffened shells']['3.6.3 Panel stiffener buckling'] = dict()

        alphaT = None
        chk4 = alphaT <= 0.6 #(3.6.2)
        checks['3.6 Longitudinally stiffened shells']['3.6.3 Panel stiffener buckling']['3.6.3.1 General'] = all(chk1, chk4)

        ##3.6.3.2 Elastic buckling strength
        alphaC = (12*(1-math.pow(v,2))) #(3.6.6)
        se = se = (fks/sigjsd) * (abs(sigxSd)/fy) #(3.6.7)

        # Table 3-3
        psi = {'Axial stress': (1+alphaC) / (1+A/(se*t)),
               'Torsion and shear stress': 5.54+1.82*math.pow(l/s, 4/3) * math.pow(alphaC, 1/3),
               'Lateral Pressure': 2*(1+math.sqrt(1+alphaC))}                      # ψ
        ksi = {'Axial stress': 0.702*Zl,
               'Torsion and shear stress': 0.856*math.pow(Zl, 3/4),
               'Lateral Pressure': 1.04*math.sqrt(Zl)}                             # ξ
        rho = {'Axial stress': 0.5,
               'Torsion and shear stress': 0.6,
               'Lateral Pressure': 0.6}

        # where A = area of one stiffener, exclusive shell plate

        ## 3.6.3.3 Effective shell width

        # --------------------3.7 Orthogonally stiffened shells--------------------------

        # --------------------3.8 Column buckling--------------------------
        checks['3.8 Column buckling'] = dict()


        k = None    #effective length factor
        ic = None   #iC = = radius of gyration of cylinder section
        Ic = None   # IC = moment of inertia of the complete cylinder section (about weakest axis), including
                    # longitudinal stiffeners/internal bulkheads if any
        Lc = None   # total cylinder length
        Ac = None   # cross sectional area of complete cylinder section;
                    # including longitudinal stiffeners/internal bulkheads if any

        chk_381 = math.pow((k*Lc) / ic, 2) >= 2.5*E/fy
        checks['3.8 Column buckling']['3.8.1 Stability requirement'] = dict()
        checks['3.8 Column buckling']['3.8.1 Stability requirement']['3.8.1'] = chk_381

        #The stability requirement for a shell-column subjected to axial compression, bending and circumferential
        # compression is given by:
        faK = None
        alpha_overline = math.sqrt(faK/fE) # (3.8.7)
        fkc = (1-0.28*math.pow(alpha_overline,2)) if alpha_overline<=1.34 \
            else 0.9 / math.pow(alpha_overline, 2) # (3.8.5) (3.8.6)
        fkcd = fkc / gammaM
        fakd = None
        sigm1Sd = None
        sigm2Sd = None
        Ic1 = None
        k1 = None
        Lc1 = None
        Ic2 = None
        k2 = None
        Lc2 = None
        fE1 = (math.pow(math.pi, 2)*E*Ic1) / ( math.pow(k1*Lc1,2) * Ac) # (3.8.3)
        fE2 = (math.pow(math.pi, 2)*E*Ic2) / ( math.pow(k2*Lc2,2) * Ac) # (3.8.3)

        chk_382 = (siga0sd/fkcd) + \
                  (1/fakd)*math.pow( math.pow( sigm1Sd/(1-siga0sd/fE1),2) +
                                     math.pow( sigm2Sd (1-siga0sd/fE2),2),0.5) <= 1 # 3.8.2
        checks['3.8 Column buckling']['3.8.1 Stability requirement']['3.8.2'] = chk_382

        ## 3.8.2 Column buckling strength

        # --------------------3.9 Torsional buckling--------------------------
        G = None
        It = None
        Ipo = None
        hs = None
        Iz = None
        lT = None

        fET = beta*(G*It/Ipo) + math.pow(math.pi, 2)* ( (E*math.pow(hs, 2)*Iz) / (Ipo*math.pow(lT,2)) )   # GENERAL (3.9.5) TODO


        alphaT_overline = math.sqrt(fy/fET) # (3.9.3)
        mu = 0.35*(alphaT_overline-0.6) #(3.9.4)

        if alphaT_overline <= 0.6:
            fT = fy  # (3.9.1)
        else:
            fT = ((1 + mu + math.pow(alphaT_overline, 2) -
                   math.sqrt(math.pow(1+mu+math.pow(alphaT_overline,2),2) -
                             4*math.pow(alphaT_overline,2)))/(2*math.pow(alphaT_overline,2)))*fy  # (3.9.2)














class CalcFatigue(Structure):
    '''
    This Class does the calculations for the plate fields. 
    Input is a structure object (getters from the Structure Class)
    '''
    def __init__(self, main_dict: dict, fatigue_dict: dict=None):
        super(CalcFatigue, self).__init__(main_dict, fatigue_dict)
        if fatigue_dict is not None:
            self._sn_curve = fatigue_dict['SN-curve']
            self._acc = fatigue_dict['Accelerations']
            self._weibull = fatigue_dict['Weibull']
            self._period = fatigue_dict['Period']
            self._k_factor = fatigue_dict['SCF']
            self._corr_loc = fatigue_dict['CorrLoc']
            self._no_of_cycles = fatigue_dict['n0']
            self._design_life = fatigue_dict['Design life']
            self._fraction = fatigue_dict['Fraction']
            self._case_order = fatigue_dict['Order']
            try:
                self._dff = fatigue_dict['DFF']
            except KeyError:
                self._dff = 2

            self.fatigue_dict = fatigue_dict

    def get_sn_curve(self):
        return self._sn_curve

    def __get_sigma_ext(self, int_press):
        return -0.5*int_press* ((self.spacing / (self.plate_th))**2) * (self._k_factor/1000**2)

    def __get_sigma_int(self, ext_press):
        return 0.5*ext_press*((self.spacing/(self.plate_th))**2) * (self._k_factor/1000**2)

    def __get_range(self, idx, int_press, ext_press):
        return 2*math.sqrt(math.pow(self.__get_sigma_ext(ext_press), 2) +
                           math.pow(self.__get_sigma_int(int_press), 2) +
                           2*self._corr_loc[idx]*self.__get_sigma_ext(ext_press)
                           *self.__get_sigma_int(int_press))

    def __get_stress_fraction(self,idx, int_press, ext_press):
        return self.__get_range(idx, int_press, ext_press) / \
               math.pow(math.log(self._no_of_cycles), 1/self._weibull[idx])

    def __get_gamma1(self,idx):
        return math.exp(gammaln(snc.get_paramter(self._sn_curve,'m1')/self._weibull[idx] + 1))

    def __get_gamma2(self,idx):
        return math.exp(gammaln(snc.get_paramter(self._sn_curve, 'm2') / self._weibull[idx] + 1))

    def get_damage_slope1(self, idx, curve, int_press=0, ext_press=0):
        m1, log_a1, k, slope = snc.get_paramter(curve,'m1'), snc.get_paramter(curve,'log a1'),\
                               snc.get_paramter(curve,'k'), snc.get_paramter(curve,'slope')
        cycles = self._design_life*365*24*3600/self._period[idx]
        thk_eff = math.log10(max(1,self.plate_th/0.025)) * k
        slope_ch = math.exp( math.log( math.pow(10, log_a1-m1*thk_eff)/slope) / m1)
        gamma1 = self.__get_gamma1(idx)
        weibull = self._weibull[idx]
        stress_frac = self.__get_stress_fraction(idx, int_press, ext_press)
        # print('Internal pressure: ', int_press)
        # print('External pressure: ', ext_press)
        # finding GAMMADIST
        if stress_frac == 0:
            return 0

        x, alpha = math.pow(slope_ch/stress_frac, weibull),1 + m1/weibull
        gamma_val = gammadist.cdf(x,alpha)
        return cycles / math.pow(10, log_a1-m1*thk_eff) * math.pow(stress_frac, m1)*gamma1*(1-gamma_val)\
               *self._fraction[idx]

    def get_damage_slope2(self, idx, curve, int_press, ext_press):
        m2, log_m2, k, slope = snc.get_paramter(curve,'m2'), snc.get_paramter(curve,'log a2'),\
                               snc.get_paramter(curve,'k'), snc.get_paramter(curve,'slope')
        cycles = self._design_life*365*24*3600/self._period[idx]
        thk_eff = math.log10(max(1,self.plate_th/25)) * k
        slope_ch = math.exp( math.log( math.pow(10, log_m2-m2*thk_eff)/slope) / m2)
        gammm2 = self.__get_gamma2(idx)
        weibull = self._weibull[idx]
        stress_frac = self.__get_stress_fraction(idx, int_press, ext_press)

        # finding GAMMADIST
        if stress_frac == 0:
            return 0
        x, alpha = math.pow(slope_ch/stress_frac, weibull),1 + m2/weibull
        gamma_val = gammadist.cdf(x,alpha)

        return cycles / math.pow(10, log_m2-m2*thk_eff) * math.pow(stress_frac, m2)*gammm2*(gamma_val)\
               *self._fraction[idx]

    def get_total_damage(self, int_press=(0, 0, 0), ext_press=(0, 0, 0)):
        damage = 0

        for idx in range(3):
            if self._fraction[idx] != 0 and self._period[idx] != 0:
                damage += self.get_damage_slope1(idx,self._sn_curve, int_press[idx], ext_press[idx]) + \
                          self.get_damage_slope2(idx,self._sn_curve, int_press[idx], ext_press[idx])

        return damage

    def set_commmon_properties(self, fatigue_dict: dict):
        ''' Setting the fatiuge properties. '''
        #self._sn_curve, self.fatigue_dict['SN-curve'] = fatigue_dict['SN-curve'], fatigue_dict['SN-curve']
        self._acc, self.fatigue_dict['Accelerations'] = fatigue_dict['Accelerations'], fatigue_dict['Accelerations']
        #self._weibull, self.fatigue_dict['Weibull'] = fatigue_dict['Weibull'], fatigue_dict['Weibull']
        #self._period, self.fatigue_dict['Period'] = fatigue_dict['Period'], fatigue_dict['Period']
        #self._k_factor, self.fatigue_dict['SCF'] = fatigue_dict['SCF'], fatigue_dict['SCF']
        #self._corr_loc, self.fatigue_dict['CorrLoc'] = fatigue_dict['CorrLoc'], fatigue_dict['CorrLoc']
        self._no_of_cycles, self.fatigue_dict['n0'] = fatigue_dict['n0'], fatigue_dict['n0']
        self._design_life, self.fatigue_dict['Design life'] = fatigue_dict['Design life'], fatigue_dict['Design life']
        self._fraction, self.fatigue_dict['Fraction'] = fatigue_dict['Fraction'], fatigue_dict['Fraction']
        #self._case_order, self.fatigue_dict['Order'] = fatigue_dict['Order'], fatigue_dict['Order']
        self._dff, self.fatigue_dict['DFF'] = fatigue_dict['DFF'], fatigue_dict['DFF']


    def set_fatigue_properties(self, fatigue_dict: dict):
        ''' Setting the fatiuge properties. '''
        self._sn_curve, self.fatigue_dict['SN-curve'] = fatigue_dict['SN-curve'], fatigue_dict['SN-curve']
        self._acc, self.fatigue_dict['Accelerations'] = fatigue_dict['Accelerations'], fatigue_dict['Accelerations']
        self._weibull, self.fatigue_dict['Weibull'] = fatigue_dict['Weibull'], fatigue_dict['Weibull']
        self._period, self.fatigue_dict['Period'] = fatigue_dict['Period'], fatigue_dict['Period']
        self._k_factor, self.fatigue_dict['SCF'] = fatigue_dict['SCF'], fatigue_dict['SCF']
        self._corr_loc, self.fatigue_dict['CorrLoc'] = fatigue_dict['CorrLoc'], fatigue_dict['CorrLoc']
        self._no_of_cycles, self.fatigue_dict['n0'] = fatigue_dict['n0'], fatigue_dict['n0']
        self._design_life, self.fatigue_dict['Design life'] = fatigue_dict['Design life'], fatigue_dict['Design life']
        self._fraction, self.fatigue_dict['Fraction'] = fatigue_dict['Fraction'], fatigue_dict['Fraction']
        self._case_order, self.fatigue_dict['Order'] = fatigue_dict['Order'], fatigue_dict['Order']
        self._dff, self.fatigue_dict['DFF'] = fatigue_dict['DFF'], fatigue_dict['DFF']

    def get_fatigue_properties(self):
        ''' Returning properties as a dictionary '''
        return self.fatigue_dict

    def get_accelerations(self):
        ''' Returning tuple of accelerattions.'''
        return self._acc

    def get_dff(self):
        return self._dff

    def get_design_life(self):
        return self._design_life

class PULSpanel():
    '''
    Takes care of puls runs
    '''
    def __init__(self, run_dict: dict = {}, puls_acceptance: float = 0.87, puls_sheet_location: str = None):
        super(PULSpanel, self).__init__()

        self._all_to_run = run_dict
        self._run_results = {}
        self._puls_acceptance = puls_acceptance
        self._puls_sheet_location = puls_sheet_location
        self._all_uf = {'buckling': list(), 'ultimate': list()}

    @property
    def all_uf(self):
        return self._all_uf

    @property
    def puls_acceptance(self):
        return self._puls_acceptance

    @puls_acceptance.setter
    def puls_acceptance(self, val):
        self._puls_acceptance = val

    @property
    def puls_sheet_location(self):
        return self._puls_sheet_location

    @puls_sheet_location.setter
    def puls_sheet_location(self, val):
        self._puls_sheet_location = val

    def set_all_to_run(self, val):
        self._all_to_run = val

    def get_all_to_run(self):
        return self._all_to_run

    def get_run_results(self):
        return self._run_results

    def set_run_results(self, val):
        self._run_results = val
        for key in self._run_results.keys():
            if any([key == 'sheet location',type(self._run_results[key]['Buckling strength']) != dict,
                    type(self._run_results[key]['Ultimate capacity']) != dict]): # TODO CHECK
                continue

            if all([type(self._run_results[key]['Buckling strength']['Actual usage Factor'][0]) == float,
                    type(self._run_results[key]['Ultimate capacity']['Actual usage Factor'][0]) == float]):
                self._all_uf['buckling'].append(self._run_results[key]['Buckling strength']['Actual usage Factor'][0])
                self._all_uf['ultimate'].append(self._run_results[key]['Ultimate capacity']['Actual usage Factor'][0])
        self._all_uf['buckling'] = np.unique(self._all_uf['buckling']).tolist()
        self._all_uf['ultimate'] = np.unique(self._all_uf['ultimate']).tolist()

    def run_all(self, store_results = False):
        '''
        Returning following results.:

        Identification:  name of line/run
        Plate geometry:       dict_keys(['Length of panel', 'Stiffener spacing', 'Plate thick.'])
        Primary stiffeners: dict_keys(['Number of stiffeners', 'Stiffener type', 'Stiffener boundary', 'Stiff. Height',
                            'Web thick.', 'Flange width', 'Flange thick.', 'Flange ecc.', 'Tilt angle'])
        Secondary stiffeners. dict_keys(['Number of sec. stiffeners', 'Secondary stiffener type', 'Stiffener boundary',
                            'Stiff. Height', 'Web thick.', 'Flange width', 'Flange thick.'])
        Model imperfections. dict_keys(['Imp. level', 'Plate', 'Stiffener', 'Stiffener tilt'])
        Material: dict_keys(['Modulus of elasticity', "Poisson's ratio", 'Yield stress plate', 'Yield stress stiffener'])
        Aluminium prop: dict_keys(['HAZ pattern', 'HAZ red. factor'])
        Applied loads: dict_keys(['Axial stress', 'Trans. stress', 'Shear stress', 'Pressure (fixed)'])
        Bound cond.: dict_keys(['In-plane support'])
        Global elastic buckling: dict_keys(['Axial stress', 'Trans. Stress', 'Trans. stress', 'Shear stress'])
        Local elastic buckling: dict_keys(['Axial stress', 'Trans. Stress', 'Trans. stress', 'Shear stress'])
        Ultimate capacity: dict_keys(['Actual usage Factor', 'Allowable usage factor', 'Status'])
        Failure modes: dict_keys(['Plate buckling', 'Global stiffener buckling', 'Torsional stiffener buckling',
                            'Web stiffener buckling'])
        Buckling strength: dict_keys(['Actual usage Factor', 'Allowable usage factor', 'Status'])
        Local geom req (PULS validity limits): dict_keys(['Plate slenderness', 'Web slend', 'Web flange ratio',
                            'Flange slend ', 'Aspect ratio'])
        CSR-Tank requirements (primary stiffeners): dict_keys(['Plating', 'Web', 'Web-flange', 'Flange', 'stiffness'])

        :return:
        '''
        import ANYstructure_local.excel_inteface as pulsxl

        iterator = self._all_to_run

        newfile = self._puls_sheet_location

        my_puls = pulsxl.PulsExcel(newfile, visible=False)
        #my_puls.set_multiple_rows(20, iterator)
        run_sp, run_up = my_puls.set_multiple_rows_batch(iterator)
        my_puls.calculate_panels(sp=run_sp, up=run_up)
        #all_results = my_puls.get_all_results()
        all_results = my_puls.get_all_results_batch(sp = run_sp, up=run_up)

        for id, data in all_results.items():
            self._run_results[id] = data

        my_puls.close_book(save=False)

        self._all_uf = {'buckling': list(), 'ultimate': list()}
        for key in self._run_results.keys():
            try:
                if all([type(self._run_results[key]['Buckling strength']['Actual usage Factor'][0]) == float,
                        type(self._run_results[key]['Ultimate capacity']['Actual usage Factor'][0]) == float]):
                    self._all_uf['buckling'].append(self._run_results[key]['Buckling strength']
                                                    ['Actual usage Factor'][0])
                    self._all_uf['ultimate'].append(self._run_results[key]['Ultimate capacity']
                                                    ['Actual usage Factor'][0])
            except TypeError:
                print('Got a type error. Life will go on. Key for PULS run results was', key)
                print(self._run_results[key])
        self._all_uf['buckling'] = np.unique(self._all_uf['buckling']).tolist()
        self._all_uf['ultimate'] = np.unique(self._all_uf['ultimate']).tolist()
        if store_results:
            store_path = os.path.dirname(os.path.abspath(__file__))+'\\PULS\\Result storage\\'
            with open(store_path+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_UP.json', 'w') as file:
                file.write(json.dumps(all_results, ensure_ascii=False))
        return all_results

    def get_utilization(self, line, method, acceptance = 0.87):
        if line in self._run_results.keys():
            if method == 'buckling':
                if type(self._run_results[line]['Buckling strength']['Actual usage Factor'][0]) == str or \
                        self._run_results[line]['Buckling strength']['Actual usage Factor'][0] is None:
                    return None
                return self._run_results[line]['Buckling strength']['Actual usage Factor'][0]/acceptance
            else:
                if type(self._run_results[line]['Ultimate capacity']['Actual usage Factor'][0]) == str or \
                        self._run_results[line]['Buckling strength']['Actual usage Factor'][0] is None:
                    return None
                return self._run_results[line]['Ultimate capacity']['Actual usage Factor'][0]/acceptance
        else:
            return None

    # def run_all_multi(self):
    #
    #     tasks = []
    #
    #     if len(self._all_to_run) > 20:
    #         processes = 10#max(cpu_count() - 1, 1)
    #
    #         def chunks(data, SIZE=10000):
    #             it = iter(data)
    #             for i in range(0, len(data), SIZE):
    #                 yield {k: data[k] for k in islice(it, SIZE)}
    #
    #         # Sample run:
    #
    #         for item in chunks({key: value for key, value in ex.run_dict.items()}, int(len(self._all_to_run)/processes)):
    #             tasks.append(item)
    #     else:
    #         tasks.append(self._all_to_run)
    #     # [print(task) for task in tasks]
    #     # print(self._all_to_run)
    #     # quit()
    #     queue = multiprocessing.SimpleQueue()
    #
    #     for idx, name in enumerate(tasks):
    #         p = Process(target=self.run_all_multi_sub, args=(name, queue, idx+1))
    #         p.start()
    #     p.join()
    #     for task in tasks:
    #         print(queue.get())

    # def run_all_multi_sub(self, iterator, queue = None, idx = 0):
    #     '''
    #     Returning following results.:
    #
    #     Identification:  name of line/run
    #     Plate geometry:       dict_keys(['Length of panel', 'Stiffener spacing', 'Plate thick.'])
    #     Primary stiffeners: dict_keys(['Number of stiffeners', 'Stiffener type', 'Stiffener boundary', 'Stiff. Height',
    #                         'Web thick.', 'Flange width', 'Flange thick.', 'Flange ecc.', 'Tilt angle'])
    #     Secondary stiffeners. dict_keys(['Number of sec. stiffeners', 'Secondary stiffener type', 'Stiffener boundary',
    #                         'Stiff. Height', 'Web thick.', 'Flange width', 'Flange thick.'])
    #     Model imperfections. dict_keys(['Imp. level', 'Plate', 'Stiffener', 'Stiffener tilt'])
    #     Material: dict_keys(['Modulus of elasticity', "Poisson's ratio", 'Yield stress plate', 'Yield stress stiffener'])
    #     Aluminium prop: dict_keys(['HAZ pattern', 'HAZ red. factor'])
    #     Applied loads: dict_keys(['Axial stress', 'Trans. stress', 'Shear stress', 'Pressure (fixed)'])
    #     Bound cond.: dict_keys(['In-plane support'])
    #     Global elastic buckling: dict_keys(['Axial stress', 'Trans. Stress', 'Trans. stress', 'Shear stress'])
    #     Local elastic buckling: dict_keys(['Axial stress', 'Trans. Stress', 'Trans. stress', 'Shear stress'])
    #     Ultimate capacity: dict_keys(['Actual usage Factor', 'Allowable usage factor', 'Status'])
    #     Failure modes: dict_keys(['Plate buckling', 'Global stiffener buckling', 'Torsional stiffener buckling',
    #                         'Web stiffener buckling'])
    #     Buckling strength: dict_keys(['Actual usage Factor', 'Allowable usage factor', 'Status'])
    #     Local geom req (PULS validity limits): dict_keys(['Plate slenderness', 'Web slend', 'Web flange ratio',
    #                         'Flange slend ', 'Aspect ratio'])
    #     CSR-Tank requirements (primary stiffeners): dict_keys(['Plating', 'Web', 'Web-flange', 'Flange', 'stiffness'])
    #
    #     :return:
    #     '''
    #     old_file = os.path.dirname(os.path.abspath(__file__))+'\\PULS\\PulsExcel_new - Copy (1).xlsm'
    #     new_file = os.path.dirname(os.path.abspath(__file__))+'\\PULS\\PulsExcel_new - Copy multi ('+str(idx)+').xlsm'
    #     shutil.copy(old_file, new_file)
    #     #time.sleep(idx*5)
    #     pythoncom.CoInitialize()
    #
    #     my_puls = pulsxl.PulsExcel(new_file, visible=False)
    #     try:
    #         my_puls.set_multiple_rows_batch(20, iterator)
    #         my_puls.calculate_panels()
    #         all_results = my_puls.get_all_results_batch()
    #         my_puls.close_book(save=True)
    #         queue.put(all_results)
    #         os.remove(new_file)
    #     except (BaseException, AttributeError):
    #         my_puls.close_book(save=False)
    #         queue.put(None)

    def get_puls_line_results(self, line):
        if line not in self._run_results.keys():
            return None
        else:
            return self._run_results[line]

    def get_string(self, line, uf = 0.87):
        '''
        :param line:
        :return:
        '''

        results = self._run_results[line]
        loc_geom = 'Ok' if all([val[0] == 'Ok' for val in results['Local geom req (PULS validity limits)']
                              .values()]) else 'Not ok'
        csr_geom = 'Ok' if all([val[0] == 'Ok' for val in results['CSR-Tank requirements (primary stiffeners)']
                              .values()]) else 'Not ok'

        ret_str = 'PULS results\n\n' +\
                  'Ultimate capacity usage factor:  ' + str(results['Ultimate capacity']['Actual usage Factor'][0]/uf)+'\n'+\
                  'Buckling strength usage factor:  ' + str(results['Buckling strength']['Actual usage Factor'][0]/uf)+'\n'+\
                  'Local geom req (PULS validity limits):   ' + loc_geom + '\n'+\
                  'CSR-Tank requirements (primary stiffeners):   ' + csr_geom
        return ret_str

    def result_changed(self, id):
        if id in self._run_results.keys():
            self._run_results.pop(id)

    def generate_random_results(self, batch_size: int = 1000, stf_type: str = None):
        '''
        Genrate random results based on user input.
        :return:
        '''

        '''
        Running iterator:
        run_dict_one = {'line3': {'Identification': 'line3', 'Length of panel': 4000.0, 'Stiffener spacing': 700.0,
                          'Plate thickness': 18.0, 'Number of primary stiffeners': 10, 'Stiffener type (L,T,F)': 'T',
                          'Stiffener boundary': 'C', 'Stiff. Height': 400.0, 'Web thick.': 12.0, 'Flange width': 200.0,
                          'Flange thick.': 20.0, 'Tilt angle': 0, 'Number of sec. stiffeners': 0,
                          'Modulus of elasticity': 210000.0, "Poisson's ratio": 0.3, 'Yield stress plate': 355.0,
                          'Yield stress stiffener': 355.0, 'Axial stress': 101.7, 'Trans. stress 1': 100.0,
                          'Trans. stress 2': 100.0, 'Shear stress': 5.0, 'Pressure (fixed)': 0.41261,
                          'In-plane support': 'Int'}}
        '''
        run_dict = {}

        profiles = hlp.helper_read_section_file('bulb_anglebar_tbar_flatbar.csv')
        if stf_type is not None:
            new_profiles = list()
            for stf in profiles:
                if stf['stf_type'][0] == stf_type:
                    new_profiles.append(stf)
            profiles = new_profiles
        lengths = np.arange(2000,6000,100)
        spacings = np.arange(500,900,10)
        thks = np.arange(10,25,1)
        axstress =transsress1 = transsress2 = shearstress = np.arange(-200,210,10) #np.concatenate((np.arange(-400,-200,10), np.arange(210,410,10)))

        pressures =  np.arange(0,0.45,0.01)
        now = time.time()
        yields = np.array([235,265,315,355,355,355,355,390,420,460])
        for idx in range(batch_size):
            ''' Adding 'Stiffener type (L,T,F)': self.stf_type,  'Stiffener boundary': 'C',
                'Stiff. Height': self.stf_web_height*1000, 'Web thick.': self.stf_web_thk*1000, 
                'Flange width': self.stf_flange_width*1000, 'Flange thick.': self.stf_flange_thk*1000}'''

            this_id = 'run_' + str(idx) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            this_stf = random.choice(profiles)

            if random.choice([True, False]):
                boundary = 'Int'
            else:
                boundary = random.choice(['GL', 'GT'])
            if random.choice([True, True, True, False]):
                stf_boundary = 'C'
            else:
                stf_boundary = 'S'
            #boundary = 'Int'
            #stf_boundary = 'C'


            yieldstress = np.random.choice(yields)
            if random.choice([True, True, True, False]):
                transstress1 = np.random.choice(transsress1)  # Using same value for trans1 and trans 2
                transstress2 = transstress1
            else:
                transstress1 = np.random.choice(transsress1)
                transstress2 = np.random.choice(transsress2)

            # run_dict[this_id] = {'Identification': this_id, 'Length of panel': np.random.choice(lengths),
            #                      'Stiffener spacing': np.random.choice(spacings),
            #                      'Plate thickness': np.random.choice(thks), 'Number of primary stiffeners': 10,
            #                      'Stiffener type (L,T,F)': 'F' if this_stf['stf_type'][0] == 'FB' else this_stf['stf_type'][0],
            #                      'Stiffener boundary': stf_boundary,
            #                      'Stiff. Height': this_stf['stf_web_height'][0]*1000,
            #                      'Web thick.': this_stf['stf_web_thk'][0]*1000,
            #                      'Flange width': 0 if this_stf['stf_type'][0] == 'F'
            #                      else this_stf['stf_flange_width'][0]*1000,
            #                      'Flange thick.': 0 if  this_stf['stf_type'][0] == 'F'
            #                      else this_stf['stf_flange_thk'][0]*1000,
            #                      'Tilt angle': 0, 'Number of sec. stiffeners': 0,
            #                      'Modulus of elasticity': 210000, "Poisson's ratio": 0.3,
            #                      'Yield stress plate':yieldstress, 'Yield stress stiffener': yieldstress,
            #                      'Axial stress': 0 if boundary == 'GT' else np.random.choice(axstress),
            #                      'Trans. stress 1': 0 if boundary == 'GL' else transstress1,
            #                      'Trans. stress 2': 0 if boundary == 'GL' else transstress2,
            #                      'Shear stress': np.random.choice(shearstress),
            #                      'Pressure (fixed)': 0 if stf_boundary == 'S' else np.random.choice(pressures),
            #                      'In-plane support': boundary, 'sp or up': 'SP'}

            same_ax = np.random.choice(axstress)
            lengths = np.arange(100, 6000, 100)
            spacings = np.arange(100, 26000, 100)
            thks = np.arange(10, 50, 1)
            boundary = random.choice(['GL', 'GT'])

            if np.random.choice([True,False,False,False]):
                support = ['SS','SS','SS','SS']
            elif np.random.choice([True,False,False,False]):
                support = ['CL','CL','CL','CL']
            else:
                support = [np.random.choice(['SS', 'CL']),np.random.choice(['SS', 'CL']),
                           np.random.choice(['SS', 'CL']),np.random.choice(['SS', 'CL'])]
            if np.random.choice([True,False]):
                press = 0
            else:
                press = np.random.choice(pressures)
            run_dict[this_id] = {'Identification': this_id, 'Length of plate': np.random.choice(lengths),
                                 'Width of c': np.random.choice(spacings),
                           'Plate thickness': np.random.choice(thks),
                         'Modulus of elasticity': 210000, "Poisson's ratio": 0.3,
                                 'Yield stress plate':yieldstress,
                         'Axial stress 1': 0 if boundary == 'GT' else same_ax,
                           'Axial stress 2': 0 if boundary == 'GT' else same_ax,
                           'Trans. stress 1': 0 if boundary == 'GL' else transstress1,
                         'Trans. stress 2': 0 if boundary == 'GL' else transstress2,
                           'Shear stress': np.random.choice(shearstress), 'Pressure (fixed)': press,
                                 'In-plane support': boundary,
                         'Rot left': support[0], 'Rot right': support[1],
                                 'Rot upper': support[2], 'Rot lower': support[3],
                           'sp or up': 'UP'}

        self._all_to_run = run_dict
        self.run_all(store_results=True)
        print('Time to run', batch_size, 'batches:', time.time() - now)




def f(name, queue):
    import time
    #print('hello', name)
    time.sleep(2)
    queue.put(name)


if __name__ == '__main__':
    import ANYstructure_local.example_data as ex
    # PULS = PULSpanel(ex.run_dict, puls_sheet_location=r'C:\Github\ANYstructure\ANYstructure\PULS\PulsExcel_new - Copy (1).xlsm')
    # PULS.run_all_multi()
    PULS = PULSpanel(puls_sheet_location=r'C:\Github\ANYstructure\ANYstructure_local\PULS\PulsExcel_new - generator.xlsm')
    for dummy in range(100):
        PULS.generate_random_results(batch_size=10000)
    # import ANYstructure_local.example_data as test
    # from multiprocessing import Process
    #
    # queue = multiprocessing.SimpleQueue()
    # tasks = ['a', 'b', 'c']
    # for name in tasks:
    #     p = Process(target=f, args=(name,queue))
    #     p.start()
    #
    # for task in tasks:
    #     print(queue.get())


    # print('Fatigue test: ')
    # my_test = CalcFatigue(test.obj_dict, test.fat_obj_dict)
    # print('Total damage: ',my_test.get_total_damage(int_press=(0,0,0), ext_press=(50000, 60000,0)))
    # print('')
    # print('Buckling test: ')
    #
    # my_buc = test.get_structure_calc_object()
    #
    # #print(my_buc.calculate_buckling_all(design_lat_press=100))
    # print(my_buc.calculate_slamming_plate(1000000))
    # print(my_buc.calculate_slamming_stiffener(1000000))
    # print(my_buc.get_net_effective_plastic_section_modulus())

    #my_test.get_total_damage(int_press=(0, 0, 0), ext_press=(0, 40000, 0))
    # import ANYstructure_local.example_data as ex
    # for example in [CalcScantlings(ex.obj_dict), CalcScantlings(ex.obj_dict2), CalcScantlings(ex.obj_dict_L)]:
    #     my_test = example
        # my_test = CalcScantlings(example)
        # my_test = CalcFatigue(example, test.fat_obj_dict2)
        # my_test.get_total_damage(int_press=(0, 0, 0), ext_press=(0, 40000, 0))
        # print('Total damage: ', my_test.get_total_damage(int_press=(0, 0, 0), ext_press=(0, 40000, 0)))
        # print(my_test.get_fatigue_properties())
        # pressure = 200
        # print(my_test.buckling_local_stiffener())
        # print('SHEAR CENTER: ',my_test.get_shear_center())
        # print('SECTION MOD: ',my_test.get_section_modulus())
        # print('SECTION MOD FLANGE: ', my_test.get_section_modulus()[0])
        # print('SHEAR AREA: ', my_test.get_shear_area())
        # print('PLASTIC SECTION MOD: ',my_test.get_plasic_section_modulus())
        # print('MOMENT OF INTERTIA: ',my_test.get_moment_of_intertia())
        # print('WEIGHT', my_test.get_weight())
        # print('PROPERTIES', my_test.get_structure_prop())
        # print('CROSS AREA', my_test.get_cross_section_area())
        # print()
        #
        # print('EFFICIENT MOMENT OF INTERTIA: ',my_test.get_moment_of_intertia(efficent_se=my_test.get_plate_efficent_b(
        #     design_lat_press=pressure)))
        # print('Se: ',my_test.calculate_buckling_all(design_lat_press=pressure,checked_side='s'))
        # print('Se: ', my_test.calculate_buckling_all(design_lat_press=pressure, checked_side='p'))
        # print('MINIMUM PLATE THICKNESS',my_test.get_dnv_min_thickness(pressure))
        # print('MINIMUM SECTION MOD.', my_test.get_dnv_min_section_modulus(pressure))
        # print()
