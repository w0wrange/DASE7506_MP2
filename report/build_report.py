"""Build the editable Markdown report into an A4 PDF; no model evaluation."""
import html
import json
from pathlib import Path
import re
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from pypdf import PdfReader


ROOT=Path(__file__).resolve().parent.parent


def inline(text):
    text=html.escape(text)
    text=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'`(.*?)`',r'<font name="Courier">\1</font>',text)
    return text


def main():
    record=json.loads((ROOT/'verification/retraining.json').read_text())
    text=(ROOT/'report/report.md').read_text().replace('{{retraining_seconds}}',f"{record['resources']['seconds']:.2f}")
    text=text.replace('{{retraining_status}}','all weight and normalizer arrays exactly equal; configuration equal; validation 743/798' if record['status']=='passed' else 'see the attached verification record')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=9.5,leading=13.6,spaceAfter=7,textColor=colors.HexColor('#17212b')))
    styles.add(ParagraphStyle(name='TableCustom',fontName='Helvetica',fontSize=8.6,leading=11.4))
    styles['Title'].fontName='Helvetica-Bold';styles['Title'].fontSize=20;styles['Title'].leading=24;styles['Title'].spaceAfter=12;styles['Title'].alignment=TA_LEFT
    styles['Heading2'].fontName='Helvetica-Bold';styles['Heading2'].fontSize=12;styles['Heading2'].leading=15;styles['Heading2'].textColor=colors.HexColor('#173b63');styles['Heading2'].spaceBefore=7;styles['Heading2'].spaceAfter=8;styles['Heading2'].keepWithNext=True
    flow=[];lines=text.splitlines();i=0;width=A4[0]-96
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line=='<!-- pagebreak -->':flow.append(PageBreak());i+=1;continue
        if line.startswith('# '):flow.append(Paragraph(inline(line[2:]),styles['Title']));i+=1;continue
        if line.startswith('## '):flow.append(Paragraph(inline(line[3:]),styles['Heading2']));i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                row=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',x) for x in row):rows.append(row)
                i+=1
            n=len(rows[0]);weights=([.48,.26,.26] if n==3 else [.44,.56] if n==2 else [.40,.13,.13,.20,.14])
            data=[[Paragraph(inline(x),styles['TableCustom']) for x in row] for row in rows]
            table=Table(data,colWidths=[width*w for w in weights],repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf0f7')),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#52677d')),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#aab8c7')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6)]))
            flow.extend([table,Spacer(1,9)]);continue
        paragraph=[]
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','<!--')):
            paragraph.append(lines[i].strip());i+=1
        flow.append(Paragraph(inline(' '.join(paragraph)),styles['BodyCustom']))
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#bdc9d5'));canvas.line(48,38,A4[0]-48,38)
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#566675'));canvas.drawString(48,25,'DASE7506 MP2 | 3036802038 | Experimental-history limitation disclosed')
        canvas.drawRightString(A4[0]-48,25,str(doc.page))
    path=ROOT/'report/report.pdf';doc=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=43,bottomMargin=49,title='Compact Candidate Verification with Arithmetic Trace Evidence',author='3036802038',pageCompression=1)
    doc.build(flow,onFirstPage=footer,onLaterPages=footer)
    pages=len(PdfReader(path).pages)
    if pages>10:raise ValueError(f'Report exceeds course page cap: {pages}')
    print(json.dumps({'report':str(path),'pages':pages}))


if __name__=='__main__':main()
