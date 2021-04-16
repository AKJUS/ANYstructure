#Making reports for the project

import time
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import Spacer
from reportlab.lib.styles import  ParagraphStyle
from reportlab.platypus import SimpleDocTemplate
from PIL import Image
import ANYstructure.example_data as test
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm, inch
from reportlab.pdfgen import canvas
from reportlab.platypus import Image, Paragraph, Table
from time import strftime, gmtime
import os
import ANYstructure.helper as hlp


def create_report(input_data):
    '''
    This class uses the module REPORTLAB to generate a report.
    :param line_obj_dict: 
    :type a dictionary
    :return: 
    '''

    Story = []

    file_name = "Report_current_results.pdf"
    doc = SimpleDocTemplate(file_name, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)

    # logo = "canvas_screenshot.gif"
    formatted_time = time.ctime()
    # im = Image(logo, 5 * inch, 4 * inch)
    # Story.append(im)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify',alignment=TA_LEFT))
    ptext = '<font size=12>%s</font>' % formatted_time
    Story.append(Paragraph(ptext, styles["Normal"]))
    Story.append(Spacer(1, 12))

    # Create return address
    ptext = '<font size=12>%s</font>' % 'The results for the structure is shown here'
    Story.append(Paragraph(ptext, styles["Justify"]))
    Story.append(Spacer(1, 3))
    ptext = '----------------------------------------------------------------------------------------------------------'
    Story.append(Paragraph(ptext, styles["Justify"]))
    Story.append(Spacer(5, 3))

    for line in input_data['lines'].keys():
        struc_obj = input_data['calc_structure'][line]
        fat_obj = input_data['calc_fatigue'][line]
        pressure = input_data['pressures'][line]

        ptext = '<font size=12>' + 'Results for: '+str(line) + '</font>'
        Story.append(Paragraph(ptext, styles["Justify"]))

        ptext = '<font size=10>'+'Plate thickness: '+ str(struc_obj.get_pl_thk()*1000)+ ' [mm], Stiffener spacing: '+\
                str(struc_obj.get_s()*1000)+' [mm]'+'</font>'
        Story.append(Paragraph(ptext, styles["Justify"]))

        ptext = '<font size=10>'+'Stiffener: '+ str(struc_obj.get_web_h()*1000)+ 'x' + str(struc_obj.get_web_thk()*1000) \
                + ' + ' + str(struc_obj.get_fl_w()*1000)+ 'x' + str(struc_obj.get_fl_thk()*1000)  +'</font>'
        Story.append(Paragraph(ptext, styles["Justify"]))

        ptext = '<font size=10>'+struc_obj.get_report_stresses()+'</font>'
        Story.append(Paragraph(ptext, styles["Justify"]))

        ptext = '<font size=10>'+struc_obj.get_results_for_report()+'</font>'
        Story.append(Paragraph(ptext, styles["Justify"]))

    Story.append(Spacer(2, 3))
    ptext = '<font size=12>END OF RESULTS</font>'
    Story.append(Paragraph(ptext, styles["Justify"]))
    my_canvas = canvas.Canvas(file_name)
    my_canvas.line(0,0,200,200)
    doc.build(Story, canvasmaker=my_canvas)

class LetterMaker(object):
    """"""

    def __init__(self, pdf_file, org, seconds, data):

        self.c = canvas.Canvas(pdf_file, pagesize=A4)
        self.styles = getSampleStyleSheet()
        self.width, self.height = A4
        self.organization = org
        self.seconds = seconds
        self.data = data
        self.draw_lines()

    def createDocument(self):
        """"""
        voffset = 100
        user = os.getlogin()
        time_now = strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())

        # create return address
        address = """<font size="12"><strong> ANYstructure report generator<br/></strong></font>""" + '<br/>' + \
                  """<font size="12"> User: </font>""" + '<font size="12">' + user + '</font>' + '<br/>' + '<br/>' + \
                  """<font size="12"> Time : </font>""" + '<font size="12">' + time_now + '</font>' + '<br/>'
        p = Paragraph(address, self.styles["Normal"])
        # add a logo and size it
        img_file_name = 'ANYstructure_logo.jpg'
        if os.path.isfile('images/' + img_file_name):
            file_path = 'images/' + img_file_name
        else:
            file_path = os.path.dirname(os.path.abspath(__file__)) + '/images/' + img_file_name
        logo = Image(file_path)
        logo.drawHeight = 1 * inch
        logo.drawWidth = 2.5 * inch
        data = [[p, logo]]
        table = Table(data, colWidths=4 * inch)
        table.setStyle([("VALIGN", (0, 0), (0, 0), "TOP")])
        table.wrapOn(self.c, self.width, self.height)
        table.drawOn(self.c, *self.coord(18, 50, mm))

        self.draw_lines()

        ptext = '<font size="12" color = "blue"><strong>' + "Compartments: " + '</strong></font>'
        self.createParagraph(ptext, 10, voffset + 85)
        delta = 0
        h_start = 130
        if self.data._tank_dict != {}:
            for name, obj in self.data._tank_dict.items():

                ptext = '<font size="7" color = "black">' + 'Name: '+ name + ', content: ' \
                        + obj.get_content() + '</font>'
                self.createParagraph(ptext, h_start, voffset + 100 + delta)

                delta += 3
                ptext = '<font size="7" color = "black">' + 'Min. elevation: ' + str(obj.get_lowest_elevation()) + \
                        ', Max. elevation: ' + str(obj.get_highest_elevation()) + '</font>'
                self.createParagraph(ptext, h_start, voffset + 100 + delta)

                delta += 3
                ptext = '<font size="7" color = "black">' + 'Applied overpressure: ' + str(obj.get_overpressure()) + \
                        '</font>'
                self.createParagraph(ptext, h_start, voffset + 100 + delta)

                delta += 3
                ptext = '<font size="7" color = "black">'+'(a_stat, a_dyn_loa, a_dyn_bal): ' + \
                        str(obj.get_accelerations()) + '</font>'
                self.createParagraph(ptext, h_start, voffset + 100 + delta)
                delta += 4
            try:
                self.c.drawImage('current_comps.png', 10,50, width=350, height=250)
            except OSError:
                self.c.drawImage('current_comps_NONE.png', 10, 50, width=350, height=250)

        # insert body of letter

        self.c.showPage()
        ptext = '<font size="12" color = "blue"><strong>' + "Results for defined structure: " + '</strong></font>'
        self.createParagraph(ptext, 10, 0)

        delta = 160
        vpos = 950

        for line in sorted(self.data._line_dict.keys()):
            vpos -= delta
            if line in self.data._line_to_struc.keys():
                struc_obj = self.data._line_to_struc[line][1]
                fo = self.data._line_to_struc[line][2]
                pressure = self.data.get_highest_pressure(line)['normal']/1000
                textobject = self.c.beginText()
                textobject.setTextOrigin(30,vpos)
                textobject.setFont("Helvetica-Oblique", 10)
                textobject.textLine('*********** '+line+' ***********')
                textobject.textLine('Plate thickness: '+ str(struc_obj.get_pl_thk()*1000)+ ' [mm]          '
                                                                                           'Stiffener spacing: '+
                                    str(struc_obj.get_s()*1000)+' [mm]'+ '          Span: '+
                                    str(round(struc_obj.get_span(),4))
                                    + ' [m]')
                textobject.textLine('Stiffener: '+ str(struc_obj.get_web_h()*1000)+ 'x' + str(struc_obj.get_web_thk()*1000)
                                    + ' + ' + str(struc_obj.get_fl_w()*1000)+ 'x' + str(struc_obj.get_fl_thk()*1000))

                textobject.textLine('Fixation paramters: kps: = '+str(struc_obj.get_kps())+ '  kpp = '
                                    + str(struc_obj.get_kpp())+
                                    ', Bending moment factors km1/km2/km3 (support/field/support)' + ' = '+
                                    str(int(struc_obj.get_km1()))+'/'+
                                    str(int(struc_obj.get_km2()))+'/'+
                                    str(int(struc_obj.get_km3())))
                textobject.textLine('Defined stresses [MPa]:  sigma_x = '+str(struc_obj.get_sigma_x())+
                                    '  sigma_y1 = '+ str(struc_obj.get_sigma_y1()) +
                                    '  sigma_y2 = '+ str(struc_obj.get_sigma_y2()) +
                                    '  tau_xy = ' + str(struc_obj.get_tau_xy()))
                textobject.textLine('ULS max pressure for line: '+ str(round(pressure,2)*1000)
                                    + ' [kPa]'+'      Pressure applied at: '+'plate side' if struc_obj.get_side()=='p'
                                    else 'stiffener side')

                if fo is not None:
                    textobject.textLine('Fatigue pressure [Pa]: '+' p_int:'+' loaded/ballast/part = '
                                        + str(round(self.data.get_color_and_calc_state()['pressure_fls'][line]['p_int']['loaded'],0))
                                        +'/'+str(round(self.data.get_color_and_calc_state()['pressure_fls'][line]['p_int']['ballast'],0))
                                        +'/'+str(round(self.data.get_color_and_calc_state()['pressure_fls'][line]['p_int']['part'],0))
                                        + ' p_ext:'+' loaded/ballast/part = '+
                                        str(round(self.data.get_color_and_calc_state()['pressure_fls'][line]['p_ext']['loaded'],0))
                                        +'/'+str(round(self.data.get_color_and_calc_state()['pressure_fls'][line]['p_ext']['ballast'],0))
                                        +'/'+str(round(self.data.get_color_and_calc_state()['pressure_fls'][line]['p_ext']['part'],0)))
                else:
                    textobject.textLine(' Fatigue pressure: No pressures defined')

                textobject.setFillColor('red') if self.data.get_color_and_calc_state()['colors'][line]['section'] == 'red' \
                    else textobject.setFillColor('black')

                textobject.textLine('Section modulus: '+str(int(min(self.data.get_color_and_calc_state()['section_modulus'][line]['sec_mod'])
                                                                *1000**3))+ ' [mm3]'+'  Min. section modulus: '+
                                    str(int(self.data.get_color_and_calc_state()['section_modulus'][line]['min_sec_mod']*1000**3))+' [mm3]'+
                                    ' -> ' + 'OK' if int(min(self.data.get_color_and_calc_state()['section_modulus'][line]['sec_mod'])*1000**3) >=
                                                     int(self.data.get_color_and_calc_state()['section_modulus'][line]['min_sec_mod']*1000**3)
                                    else 'Section modulus: '+str(int(min(self.data.get_color_and_calc_state()['section_modulus'][line]['sec_mod'])
                                                                     *1000**3))+ ' [mm3]'+ '  Min. section modulus: '+
                                    str(int(self.data.get_color_and_calc_state()['section_modulus'][line]['min_sec_mod']*1000**3))+' [mm3]'+
                                    ' -> ' + 'NOT OK')
                textobject.setFillColor('black')
                textobject.setFillColor('red') if self.data.get_color_and_calc_state()['colors'][line]['thickness'] == 'red' \
                    else textobject.setFillColor('black')
                textobject.textLine('Min plate thickness:  '+
                                    str(round(self.data.get_color_and_calc_state()['thickness'][line]['min_thk'],2)) + ' [mm] '
                                    ' -> ' +
                                    'OK' if struc_obj.get_pl_thk()*1000 >=
                                            self.data.get_color_and_calc_state()['thickness'][line]['min_thk'] else
                                    'Min plate thickness:  '+ str(round(
                                        self.data.get_color_and_calc_state()['thickness'][line]['min_thk'],2)) + ' [mm] '
                                    ' -> '+'NOT OK')
                textobject.setFillColor('black')
                textobject.setFillColor('red') if self.data.get_color_and_calc_state()['colors'][line]['shear'] == 'red' \
                    else textobject.setFillColor('black')
                textobject.textLine('Shear area: '+str(int(self.data.get_color_and_calc_state()['shear_area'][line]['shear_area']*1000**2))+' [mm2] '+
                                    '   Min shear area: '+str(int(self.data.get_color_and_calc_state()['shear_area'][line]['min_shear_area']*1000**2))
                                    + ' [mm2] ' +
                                    ' -> ' + 'OK' if self.data.get_color_and_calc_state()['shear_area'][line]['shear_area'] >=
                                                     self.data.get_color_and_calc_state()['shear_area'][line]['min_shear_area']
                                    else 'Shear area: '+str(int(self.data.get_color_and_calc_state()['shear_area'][line]['shear_area']*1000**2))+
                                         ' [mm2] ' +
                                         '   Min shear area: '+str(int(self.data.get_color_and_calc_state()['shear_area'][line]['min_shear_area']*1000**2))
                                         + ' [mm2] ' + ' -> ' + 'NOT OK')
                textobject.setFillColor('black')
                textobject.setFillColor('red') if self.data.get_color_and_calc_state()['colors'][line]['buckling'] == 'red' \
                    else textobject.setFillColor('black')
                textobject.textLine('Highest buckling utilization: '+
                                    str(round(max(self.data.get_color_and_calc_state()['buckling'][line]),2))+
                                    ' -> '+'OK' if max(self.data.get_color_and_calc_state()['buckling'][line]) < 1 else
                                    'Highest buckling utilization: '+
                                    str(round(max(self.data.get_color_and_calc_state()['buckling'][line]),2))+' -> '+'NOT OK')
                textobject.setFillColor('black')
                textobject.setFillColor('red') if self.data.get_color_and_calc_state()['colors'][line]['fatigue'] == 'red' \
                    else textobject.setFillColor('black')
                if self.data.get_color_and_calc_state()['fatigue'][line]['damage'] is not None:
                    textobject.textLine('Fatigue (plate/stiffeners) utilization: '+
                                        str(round(self.data.get_color_and_calc_state()['fatigue'][line]['damage'],2))+ ' * DFF('+
                                        str(self.data.get_color_and_calc_state()['fatigue'][line]['dff']) + ') = ' +
                                        str(round(self.data.get_color_and_calc_state()['fatigue'][line]['damage']*
                                                  self.data.get_color_and_calc_state()['fatigue'][line]['dff'],2)) + ' (SN-curve = '+
                                        self.data.get_color_and_calc_state()['fatigue'][line]['curve']+')')

                else:
                    textobject.textLine('No fatigue results')

                textobject.textLine('Utilization percentage (highest calculated): '+
                                    str(int(max(self.data.get_color_and_calc_state()['utilization'][line].values())*100))+ '%')

                textobject.setFillColor('black')
                self.c.drawText(textobject)
            else:
                textobject.setFont("Helvetica-Oblique", 10)
                textobject.textLine('*********** '+line+' ***********')
                textobject.textLine('(no structural properties defined)')
            if vpos <= 290:
                self.c.showPage()
                vpos = 950

    # ----------------------------------------------------------------------
        self.c.showPage()
        self.draw_lines(draw_type='section')
        self.c.showPage()
        self.draw_lines(draw_type='plate')
        self.c.showPage()
        self.draw_lines(draw_type='pressure')
        self.c.showPage()
        self.draw_lines(draw_type='utilization')

    def draw_lines(self, draw_type = 'UF'):
        '''
        Draw the defined lines.
        :return:
        '''
        import matplotlib
        points = self.data._point_dict
        lines = self.data._line_dict
        colors = self.data.get_color_and_calc_state()['colors']
        highest_y = max([coord[1] for coord in points.values()])
        highest_x = max([coord[0] for coord in points.values()])
        if any([highest_x == 0, highest_y == 0]):
            scale = 10
        else:
            scale = min(500/highest_y, 500/highest_x, 10)
        if draw_type == 'UF':
            origo = (50,350)
        else:
            origo = (50, 450)
        self.c.setLineWidth(2)
        self.c.setStrokeColor('red')
        idx, drawed_data = 0, list()
        for line, pt in lines.items():
            if draw_type == 'UF':
                try:
                    self.c.setStrokeColor('red' if 'red' in colors[line].values() else 'green')
                except KeyError:
                    self.c.setStrokeColor('black')
            elif draw_type == 'section':
                self.data._new_colorcode_beams.set(True)
                self.data._new_colorcode_plates.set(False)
                self.data._new_colorcode_pressure.set(False)
                line_data = self.data.draw_canvas(get_color_for_line = line)
                self.c.setStrokeColor(line_data[0])
                if line_data[1] not in drawed_data:
                    textobject = self.c.beginText()
                    if 400 - 20 * idx > 20:
                        textobject.setTextOrigin(50, 400 - 20 * idx)
                    else:
                        textobject.setTextOrigin(300, 400 - 20 * idx)
                    textobject.setFillColor(line_data[0])
                    textobject.setFont("Helvetica-Oblique", 10)
                    textobject.textLine(line_data[1])
                    self.c.drawText(textobject)
                    drawed_data.append(line_data[1])
                    idx += 1

            elif draw_type == 'plate':
                self.data._new_colorcode_beams.set(False)
                self.data._new_colorcode_plates.set(True)
                self.data._new_colorcode_pressure.set(False)
                line_data = self.data.draw_canvas(get_color_for_line=line)
                self.c.setStrokeColor(line_data[0])
                if line_data[1] not in drawed_data:
                    textobject = self.c.beginText()
                    if 400 - 20 * idx > 20:
                        textobject.setTextOrigin(50, 400 - 20 * idx)
                    else:
                        textobject.setTextOrigin(300, 400 - 20 * idx)
                    textobject.setFillColor(line_data[0])
                    textobject.setFont("Helvetica-Oblique", 10)
                    textobject.textLine(str(line_data[1]))
                    self.c.drawText(textobject)
                    drawed_data.append(line_data[1])
                    idx += 1
            elif draw_type == 'pressure':
                self.data._new_colorcode_beams.set(False)
                self.data._new_colorcode_plates.set(False)
                self.data._new_colorcode_pressure.set(True)
                line_data = self.data.draw_canvas(get_color_for_line=line)
                self.c.setStrokeColor(line_data[0])
            elif draw_type == 'utilization':
                from matplotlib import pyplot as plt
                import matplotlib
                cmap_sections = plt.get_cmap('jet')
                all_utils = [max(list(self.data.get_color_and_calc_state()['utilization'][line].values()))
                             for line in self.data._line_to_struc.keys()]
                this_util = max(list(self.data.get_color_and_calc_state()['utilization'][line].values()))
                norm_uf = this_util/(max(all_utils))
                self.c.setStrokeColor(matplotlib.colors.rgb2hex(cmap_sections(norm_uf)))


            x1, y1 = points['point'+str(pt[0])][0] * scale + origo[0], \
                     points['point'+str(pt[0])][1] * scale + origo[1]
            x2, y2 = points['point'+str(pt[1])][0] * scale + origo[0], \
                     points['point'+str(pt[1])][1] * scale + origo[1]
            self.c.line(x1,y1,x2,y2)

            textobject = self.c.beginText()
            textobject.setTextOrigin(x1+(x2-x1)*0.5-5, y1 + (y2-y1)*0.5+2 )
            textobject.setFont("Helvetica-Oblique", 9)
            textobject.textLine(str(hlp.get_num(line)))
            self.c.drawText(textobject)


        if draw_type == 'UF':
            pass
        elif draw_type == 'section':
            textobject = self.c.beginText()
            textobject.setTextOrigin(50,800)
            textobject.setFont("Helvetica-Oblique", 15)
            textobject.setFillColor('black')
            textobject.textLine('Model beam section properties')
            self.c.drawText(textobject)

        elif draw_type == 'plate':
            textobject = self.c.beginText()
            textobject.setTextOrigin(50, 800)
            textobject.setFillColor('black')
            textobject.setFont("Helvetica-Oblique", 15)
            textobject.textLine('Model plate thicknesses')
            self.c.drawText(textobject)

        elif draw_type == 'pressure':
            textobject = self.c.beginText()
            textobject.setTextOrigin(50, 800)
            textobject.setFillColor('black')
            textobject.setFont("Helvetica-Oblique", 15)
            textobject.textLine('Highest pressures for lines in model')
            self.c.drawText(textobject)
            idx = 0
            from matplotlib import pyplot as plt
            cmap_sections = plt.get_cmap('jet')
            for press in line_data[1]:
                textobject = self.c.beginText()
                if 400 - 20 * idx > 20:
                    textobject.setTextOrigin(50, 400 - 20 * idx)
                else:
                    textobject.setTextOrigin(300, 400 - 20 * idx)

                textobject.setFillColor(matplotlib.colors.rgb2hex(cmap_sections(press/max(line_data[1]))))
                textobject.setFont("Helvetica-Oblique", 10)
                textobject.textLine(str(press))
                self.c.drawText(textobject)
                drawed_data.append(line_data[1])
                idx += 1
        elif draw_type == 'utilization':
            textobject = self.c.beginText()
            textobject.setTextOrigin(50, 800)
            textobject.setFillColor('black')
            textobject.setFont("Helvetica-Oblique", 15)
            textobject.textLine('Utilization factors (max of all checks)')
            self.c.drawText(textobject)
            import numpy as np
            from matplotlib import pyplot as plt
            cmap_sections = plt.get_cmap('jet')
            for idx, uf in enumerate(np.arange(0, 1.1, 0.1)):
                textobject = self.c.beginText()
                if 400 - 20 * idx > 20:
                    textobject.setTextOrigin(50, 400 - 20 * idx)
                else:
                    textobject.setTextOrigin(300, 400 - 20 * idx)
                textobject.setFillColor(matplotlib.colors.rgb2hex(cmap_sections(uf)))
                textobject.setFont("Helvetica-Oblique", 10)
                textobject.textLine(str('UF = ' +str(round(uf,1))))
                self.c.drawText(textobject)

    def coord(self, x, y, unit=1):
        """
        # http://stackoverflow.com/questions/4726011/wrap-text-in-a-table-reportlab
        Helper class to help position flowables in Canvas objects
        """
        x, y = x * unit, self.height - y * unit
        return x, y

        # ----------------------------------------------------------------------

    def createParagraph(self, ptext, x, y, style=None):
        """"""
        if not style:
            style = self.styles["Normal"]
        p = Paragraph(ptext, style=style)
        p.wrapOn(self.c, self.width, self.height)
        p.drawOn(self.c, *self.coord(x, y, mm))

    # ----------------------------------------------------------------------
    def savePDF(self):
        """"""
        self.c.save()

    # ----------------------------------------------------------------------

if __name__ == '__main__':
    import multiprocessing, ctypes, tkinter
    import ANYstructure.main_application as app
    multiprocessing.freeze_support()
    errorCode = ctypes.windll.shcore.SetProcessDpiAwareness(2)
    root = tkinter.Tk()
    my_app = app.Application(root)
    ship_example = r'C:\Github\ANYstructure\ANYstructure\ship_section_example.txt'
    my_app.openfile(ship_example)
    my_app.report_generate(autosave=True)
    # doc = LetterMaker("example.pdf", "The MVP", 10, to_report_gen)
    # doc.createDocument()
    # doc.savePDF()

