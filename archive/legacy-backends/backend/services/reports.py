"""
Clinical Report Generation Service
===================================

Generates comprehensive PDF clinical reports with risk assessment, 
predictions, explanations, and clinical recommendations.

Features:
- Professional clinical report formatting
- Risk score visualization
- Feature importance tables
- Clinical summary and recommendations
- Patient demographics
- Longitudinal trends
- Multi-page reports with headers/footers
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import HexColor, black, white, red, green, yellow
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether, Preformatted, Frame, PageTemplate
)
from reportlab.pdfgen import canvas
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ClinicalReportGenerator:
    """
    Generates professional clinical risk assessment reports in PDF format.
    
    Combines patient data, model predictions, SHAP explanations, and clinical
    recommendations into a comprehensive report.
    """
    
    def __init__(self, output_path: Optional[str] = None):
        """
        Initialize report generator.
        
        Args:
            output_path: Path to save PDF (if None, returns bytes)
        """
        self.output_path = output_path
        self.page_size = letter
        self.margin = 0.5 * inch
        self.styles = self._create_styles()
        self.colors = {
            "high_risk": HexColor("#DC143C"),      # Crimson
            "moderate_risk": HexColor("#FF8C00"),  # Dark Orange
            "low_risk": HexColor("#28A745"),       # Green
            "header": HexColor("#003366"),         # Navy
            "subheader": HexColor("#4A90E2"),      # Blue
            "border": HexColor("#CCCCCC"),         # Light Gray
        }
    
    def _create_styles(self) -> Dict:
        """
        Create custom paragraph styles for report.
        
        Returns:
            Dictionary of ParagraphStyle objects
        """
        styles = getSampleStyleSheet()
        
        # Title style
        styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=self.colors["header"],
            spaceAfter=30,
            alignment=TA_CENTER,
            bold=True
        ))
        
        # Section header style
        styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=self.colors["subheader"],
            spaceAfter=12,
            spaceBefore=12,
            bold=True,
            borderPadding=5
        ))
        
        # Risk score style
        styles.add(ParagraphStyle(
            name='RiskScore',
            parent=styles['Normal'],
            fontSize=16,
            bold=True,
            spaceAfter=6
        ))
        
        # Clinical recommendation style
        styles.add(ParagraphStyle(
            name='Recommendation',
            parent=styles['Normal'],
            fontSize=11,
            textColor=HexColor("#333333"),
            spaceAfter=6,
            leftIndent=20
        ))
        
        return styles
    
    def _generate_header_footer(self, canvas_obj, doc):
        """
        Add header and footer to each page.
        
        Args:
            canvas_obj: ReportLab canvas
            doc: SimpleDocTemplate instance
        """
        # Header
        canvas_obj.setFont("Helvetica-Bold", 12)
        canvas_obj.drawString(
            self.margin,
            self.page_size[1] - 0.4 * inch,
            "CLINICAL RISK ASSESSMENT REPORT"
        )
        
        # Footer
        canvas_obj.setFont("Helvetica", 9)
        canvas_obj.drawRightString(
            self.page_size[0] - self.margin,
            0.5 * inch,
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')} | Page {doc.page}"
        )
    
    def generate_report(
        self,
        patient_data: Dict[str, Any],
        prediction_data: Dict[str, Any],
        explanation_data: Dict[str, Any],
        clinical_summary: Optional[Dict[str, Any]] = None,
        trend_data: Optional[Dict[str, Any]] = None,
        images: Optional[Dict[str, bytes]] = None
    ) -> Optional[bytes]:
        """
        Generate complete clinical report.
        
        Args:
            patient_data: Patient demographics and history
            prediction_data: Model prediction results
            explanation_data: SHAP explanation details
            clinical_summary: Clinical interpretation
            trend_data: Historical trend information
            images: Dictionary of image names to PNG bytes
            
        Returns:
            PDF bytes or None if error
        """
        try:
            # Create BytesIO buffer or file
            if self.output_path:
                buffer = open(self.output_path, 'wb')
            else:
                buffer = io.BytesIO()
            
            # Create document
            doc = SimpleDocTemplate(
                buffer,
                pagesize=self.page_size,
                rightMargin=self.margin,
                leftMargin=self.margin,
                topMargin=1.2 * inch,
                bottomMargin=0.8 * inch,
                title="Clinical Risk Assessment Report"
            )
            
            # Build story (content)
            story = []
            
            # Add title
            story.append(self._create_title_section())
            story.append(Spacer(1, 0.3 * inch))
            
            # Add patient information
            story.append(self._create_patient_section(patient_data))
            story.append(Spacer(1, 0.2 * inch))
            
            # Add prediction summary
            story.append(self._create_prediction_summary_section(
                prediction_data,
                clinical_summary
            ))
            story.append(Spacer(1, 0.2 * inch))
            
            # Add feature importance / explanation
            story.append(self._create_explanation_section(explanation_data))
            story.append(Spacer(1, 0.2 * inch))
            
            # Add clinical recommendations
            if clinical_summary:
                story.append(self._create_recommendations_section(clinical_summary))
                story.append(Spacer(1, 0.2 * inch))
            
            # Add trends if available
            if trend_data:
                story.append(PageBreak())
                story.append(self._create_trends_section(trend_data))
                story.append(Spacer(1, 0.2 * inch))
            
            # Add visualizations if available
            if images:
                story.append(PageBreak())
                story.append(self._create_visualizations_section(images))
                story.append(Spacer(1, 0.2 * inch))
            
            # Add disclaimer
            story.append(Spacer(1, 0.3 * inch))
            story.append(self._create_disclaimer_section())
            
            # Build PDF
            doc.build(story, onFirstPage=self._generate_header_footer, onLaterPages=self._generate_header_footer)
            
            if self.output_path:
                buffer.close()
                with open(self.output_path, 'rb') as f:
                    return f.read()
            else:
                buffer.seek(0)
                return buffer.getvalue()
        
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return None
    
    def _create_title_section(self) -> Paragraph:
        """Create report title section."""
        return Paragraph(
            "CLINICAL RISK ASSESSMENT REPORT",
            self.styles['ReportTitle']
        )
    
    def _create_patient_section(self, patient_data: Dict[str, Any]) -> KeepTogether:
        """
        Create patient information section.
        
        Args:
            patient_data: Patient demographics
            
        Returns:
            KeepTogether containing patient info
        """
        content = []
        content.append(Paragraph("PATIENT INFORMATION", self.styles['SectionHeader']))
        
        # Patient info table
        patient_info = [
            ["Patient ID:", patient_data.get("patient_id", "N/A")],
            ["Name:", patient_data.get("name", "N/A")],
            ["Date of Birth:", patient_data.get("date_of_birth", "N/A")],
            ["Age:", patient_data.get("age", "N/A")],
            ["Gender:", patient_data.get("gender", "N/A")],
            ["Clinical Diagnosis:", patient_data.get("diagnosis", "N/A")],
        ]
        
        table = Table(patient_info, colWidths=[2.5 * inch, 3.5 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), HexColor("#F0F0F0")),
            ('TEXTCOLOR', (0, 0), (-1, -1), black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, self.colors["border"]),
        ]))
        
        content.append(table)
        return KeepTogether(content)
    
    def _create_prediction_summary_section(
        self,
        prediction_data: Dict[str, Any],
        clinical_summary: Optional[Dict[str, Any]] = None
    ) -> KeepTogether:
        """
        Create prediction and risk assessment section.
        
        Args:
            prediction_data: Model prediction results
            clinical_summary: Clinical interpretation
            
        Returns:
            KeepTogether containing prediction summary
        """
        content = []
        content.append(Paragraph("RISK ASSESSMENT", self.styles['SectionHeader']))
        
        risk_score = prediction_data.get("prediction_score", 0)
        risk_level = clinical_summary.get("risk_level", "UNKNOWN") if clinical_summary else "UNKNOWN"
        
        # Determine color
        if risk_level == "HIGH":
            color = self.colors["high_risk"]
        elif risk_level == "MODERATE":
            color = self.colors["moderate_risk"]
        else:
            color = self.colors["low_risk"]
        
        # Risk score display
        risk_text = f"Risk Score: {risk_score:.1%} | Risk Level: {risk_level}"
        risk_para = Paragraph(
            f'<font color="{color.hexval()}" size=14><b>{risk_text}</b></font>',
            self.styles['Normal']
        )
        content.append(risk_para)
        content.append(Spacer(1, 0.1 * inch))
        
        # Risk description
        if clinical_summary:
            description = clinical_summary.get("risk_description", "")
            if description:
                content.append(Paragraph(description, self.styles['Normal']))
        
        return KeepTogether(content)
    
    def _create_explanation_section(self, explanation_data: Dict[str, Any]) -> KeepTogether:
        """
        Create feature importance explanation section.
        
        Args:
            explanation_data: SHAP explanation details
            
        Returns:
            KeepTogether containing explanation
        """
        content = []
        content.append(Paragraph("KEY CONTRIBUTING FACTORS", self.styles['SectionHeader']))
        
        top_factors = explanation_data.get("feature_contributions", [])[:10]
        
        if not top_factors:
            content.append(Paragraph("No contributing factors available.", self.styles['Normal']))
            return KeepTogether(content)
        
        # Create factors table
        factors_data = [["Feature", "Impact Direction", "Magnitude", "Current Value"]]
        
        for factor in top_factors:
            direction = "↑ Increases" if factor.get("contribution", 0) > 0 else "↓ Decreases"
            magnitude = f"{abs(factor.get('contribution', 0)):.4f}"
            value = f"{factor.get('value', 0):.2f}"
            
            factors_data.append([
                factor.get("feature", "Unknown"),
                direction,
                magnitude,
                value
            ])
        
        table = Table(factors_data, colWidths=[2 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.colors["subheader"]),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor("#F9F9F9")),
            ('GRID', (0, 0), (-1, -1), 0.5, self.colors["border"]),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor("#F9F9F9")]),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        
        content.append(table)
        return KeepTogether(content)
    
    def _create_recommendations_section(self, clinical_summary: Dict[str, Any]) -> KeepTogether:
        """
        Create clinical recommendations section.
        
        Args:
            clinical_summary: Clinical interpretation with recommendations
            
        Returns:
            KeepTogether containing recommendations
        """
        content = []
        content.append(Paragraph("CLINICAL RECOMMENDATIONS", self.styles['SectionHeader']))
        
        recommendations = clinical_summary.get("recommendations", [])
        
        for rec in recommendations:
            content.append(Paragraph(f"• {rec}", self.styles['Recommendation']))
        
        return KeepTogether(content)
    
    def _create_trends_section(self, trend_data: Dict[str, Any]) -> KeepTogether:
        """
        Create longitudinal trends section.
        
        Args:
            trend_data: Historical trend information
            
        Returns:
            KeepTogether containing trends
        """
        content = []
        content.append(Paragraph("LONGITUDINAL TRENDS", self.styles['SectionHeader']))
        
        trend_direction = trend_data.get("trend_direction", "stable")
        avg_risk = trend_data.get("average_risk", 0)
        max_risk = trend_data.get("max_risk", 0)
        min_risk = trend_data.get("min_risk", 0)
        
        trends_text = f"""
        <b>Trend Analysis ({trend_data.get('days_analyzed', 30)}-day period):</b><br/>
        Overall Trend: <b>{trend_direction.upper()}</b><br/>
        Average Risk: {avg_risk:.2%}<br/>
        Maximum Risk: {max_risk:.2%}<br/>
        Minimum Risk: {min_risk:.2%}
        """
        
        content.append(Paragraph(trends_text, self.styles['Normal']))
        
        return KeepTogether(content)
    
    def _create_visualizations_section(self, images: Dict[str, bytes]) -> KeepTogether:
        """
        Create visualizations section with SHAP plots.
        
        Args:
            images: Dictionary of image names to PNG bytes
            
        Returns:
            KeepTogether containing visualizations
        """
        content = []
        content.append(Paragraph("VISUAL ANALYSIS", self.styles['SectionHeader']))
        
        for image_name, image_bytes in images.items():
            try:
                # Convert bytes to Image
                image_buffer = io.BytesIO(image_bytes)
                img = Image(image_buffer, width=6 * inch, height=4 * inch)
                content.append(img)
                content.append(Spacer(1, 0.2 * inch))
                content.append(Paragraph(image_name.title(), self.styles['Normal']))
                content.append(Spacer(1, 0.3 * inch))
            except Exception as e:
                logger.error(f"Error adding image {image_name}: {str(e)}")
        
        return KeepTogether(content)
    
    def _create_disclaimer_section(self) -> KeepTogether:
        """
        Create legal disclaimer section.
        
        Returns:
            KeepTogether containing disclaimer
        """
        disclaimer_text = """
        <b>DISCLAIMER:</b><br/>
        <font size=8>
        This report is generated by an automated machine learning model and is intended for clinical 
        decision support only. It should not replace clinical judgment or professional medical review. 
        All predictions should be validated by qualified healthcare professionals. The model's accuracy 
        and applicability should be verified for the specific clinical context. This report is confidential 
        and intended for authorized personnel only.
        </font>
        """
        
        content = [Paragraph(disclaimer_text, self.styles['Normal'])]
        return KeepTogether(content)
    
    def generate_batch_reports(
        self,
        patient_records: List[Dict[str, Any]],
        output_dir: str
    ) -> Tuple[int, int]:
        """
        Generate reports for multiple patients.
        
        Args:
            patient_records: List of patient data dictionaries
            output_dir: Directory to save PDF reports
            
        Returns:
            Tuple of (successful, failed) report counts
        """
        successful = 0
        failed = 0
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for record in patient_records:
            try:
                patient_id = record.get("patient_data", {}).get("patient_id", "unknown")
                filename = output_path / f"report_{patient_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
                
                generator = ClinicalReportGenerator(str(filename))
                result = generator.generate_report(
                    patient_data=record.get("patient_data", {}),
                    prediction_data=record.get("prediction_data", {}),
                    explanation_data=record.get("explanation_data", {}),
                    clinical_summary=record.get("clinical_summary"),
                    trend_data=record.get("trend_data"),
                    images=record.get("images")
                )
                
                if result:
                    successful += 1
                    logger.info(f"Report generated: {filename}")
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Error generating report for patient {patient_id}: {str(e)}")
                failed += 1
        
        return successful, failed
