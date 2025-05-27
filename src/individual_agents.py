"""
Document Forgery Detection System - Individual Analysis Agents
Specialized agents for different types of document inconsistencies

Each agent focuses on a specific type of document forgery indicator:
1. Digital Artifacts Agent - detects image manipulation artifacts
2. Font Inconsistencies Agent - identifies font and typography issues  
3. Date Verification Agent - checks date consistency and validity
4. Layout Issues Agent - detects alignment and spacing problems
5. Design Verification Agent - validates logos, branding, and design elements
6. Content Analysis Agent - examines content logic and mathematical accuracy

All agents use OpenAI models for sophisticated document analysis.
"""

import base64
import io
import json
from datetime import datetime
from typing import Dict, List
from PIL import Image
from pdf2image import convert_from_path
import openai
from loguru import logger
import os
from dotenv import load_dotenv

from .models import IndividualAnalysisResult, ForgeryIndicator, BoundingBox

load_dotenv()

class IndividualAnalysisAgents:
    """Collection of specialized agents for document forgery detection"""
    
    def __init__(self, model_type: str = "standard"):
        """
        Initialize agents with OpenAI model preference
        
        Args:
            model_type: "reasoning" for o4-mini or "standard" for gpt-4.1
        """
        self.model_type = model_type
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.setup_apis()

    def setup_apis(self):
        """Setup OpenAI API clients"""
        try:
            # OpenAI setup
            openai_key = os.getenv('OPENAI_API_KEY')
            
            # Get model names from environment variables
            self.reasoning_model_name = os.getenv('OPENAI_MODEL_NAME1', 'o3')  # Default to o3 if not set
            self.standard_model_name = os.getenv('OPENAI_MODEL_NAME2', 'gpt-4.1')   # Default to gpt-4.1 if not set
            
            # Set current model and temperature based on model_type
            if self.model_type == "reasoning":
                self.current_model_name = self.reasoning_model_name
                self.temperature = 1  # Reasoning models only support temperature=1
            else:
                self.current_model_name = self.standard_model_name
                self.temperature = 0  # Standard models use temperature=0 for deterministic results
                
            if openai_key:
                self.openai_client = openai.OpenAI(api_key=openai_key)
                logger.info(f"OpenAI API initialized for individual agents with model: {self.current_model_name}, temperature: {self.temperature}")
            else:
                self.openai_client = None
                logger.error("OpenAI API key not found")
                raise Exception("OpenAI API key is required")
                
        except Exception as e:
            logger.error(f"Error setting up OpenAI API: {str(e)}")
            self.openai_client = None
            raise e

    def digital_artifacts_agent(self, file_path: str, number_of_pages: int = 1) -> Dict:
        """
        Analyze document for digital artifacts and tampering signs
        
        Args:
            file_path: Path to the document file
            number_of_pages: Number of pages to analyze (for PDFs)
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Running digital artifacts agent")
        
        analysis_prompt = f"""You are a specialist in detecting digital manipulation and artifacts in documents. Perform a comprehensive analysis of the provided document for signs of digital tampering, forgery, or artificial generation.

**CRITICAL ANALYSIS AREAS:**

1. **COMPRESSION ARTIFACTS & IMAGE QUALITY:**
   - Examine JPEG compression patterns - legitimate scans typically show uniform compression
   - Look for inconsistent compression levels across different regions
   - Check for unusual pixelation or blockiness that suggests re-compression
   - Verify if image quality matches expected resolution for document type
   - Identify areas with suspiciously high/low quality compared to surrounding content

2. **DIGITAL TAMPERING SIGNATURES:**
   - Detect copy-paste artifacts: repeated patterns, cloned regions, identical noise patterns
   - Look for healing brush or clone stamp tool marks
   - Identify inconsistent noise patterns across the document
   - Check for unnatural edge transitions or ghosting effects
   - Examine metadata inconsistencies if visible

3. **TEXT RENDERING ANOMALIES:**
   - Compare text rendering quality - OCR-generated vs. original text
   - Look for anti-aliasing inconsistencies in text edges
   - Check for bitmap vs. vector text quality differences
   - Identify text that appears pasted or overlaid on background
   - Examine subpixel rendering inconsistencies

4. **LOGO & WATERMARK INTEGRITY:**
   - Verify official logos for distortion, pixelation, or quality degradation
   - Check watermark authenticity and integration with document background
   - Look for security features like microprint, guilloche patterns, or special inks
   - Examine logo placement and proportions against known standards
   - Verify color accuracy and gradients in official branding elements

5. **LIGHTING & SHADOW CONSISTENCY:**
   - Analyze lighting direction and intensity across all elements
   - Check for inconsistent shadows or highlights on text/images
   - Look for elements that appear "flat" or lack natural document texture
   - Examine reflection patterns on glossy or embossed elements

6. **DIGITAL GENERATION INDICATORS:**
   - Look for AI-generated content signatures (unnatural perfection, artifacts)
   - Check for computer-generated fonts that mimic handwriting
   - Identify synthetic backgrounds or textures
   - Examine unrealistic color distributions or gradients

**DOCUMENT TYPE SPECIFIC CHECKS:**
- **Financial Documents:** Check security features, microprinting, special papers
- **Medical Records:** Verify letterhead quality, official stamps, consistent formatting
- **Legal Documents:** Examine notary seals, official signatures, court stamps
- **Identity Documents:** Check photo quality, holographic elements, RFID chips
- **Invoices/Receipts:** Verify printer quality, thermal printing characteristics

**ANALYSIS FRAMEWORK:**
Rate each area on a scale of 1-10 (1=clearly legitimate, 10=obviously forged)
- Consider cumulative evidence across all areas
- Weight findings based on document type and expected quality standards
- Account for legitimate variations in scanning/photography conditions

**OUTPUT REQUIREMENTS:**
1. **Overall Assessment:** Determine if the document is 'FAKE' or 'LEGIT'
2. **Confidence Level:** High/Medium/Low based on evidence strength
3. **Specific Findings:** List concrete digital artifacts found with exact locations
4. **Technical Details:** Describe compression patterns, quality inconsistencies, manipulation signs
5. **Risk Factors:** Identify which specific digital indicators suggest forgery

**IMPORTANT:** If marking as FAKE, your reasoning MUST include:
- Specific digital artifacts detected with precise descriptions
- Technical analysis of compression, quality, or rendering issues
- Exact text or elements showing digital manipulation
- Comparison to expected standards for this document type

Current date for temporal analysis: {self.current_date}

Analyze systematically and provide detailed technical reasoning for your assessment."""
        
        return self._analyze_with_prompt(file_path, analysis_prompt, "digital_artifacts", number_of_pages)

    def font_inconsistencies_agent(self, file_path: str, number_of_pages: int = 1) -> Dict:
        """
        Analyze document for font inconsistencies
        
        Args:
            file_path: Path to the document file
            number_of_pages: Number of pages to analyze (for PDFs)
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Running font inconsistencies agent")
        
        analysis_prompt = f"""You are a typography and font forensics expert specializing in detecting document forgeries through font analysis. Perform a meticulous examination of all textual elements for inconsistencies that indicate tampering or forgery.

**COMPREHENSIVE FONT ANALYSIS FRAMEWORK:**

1. **FONT FAMILY & TYPEFACE CONSISTENCY:**
   - Identify all font families used throughout the document
   - Verify consistency within similar content sections (headers, body text, footnotes)
   - Check for inappropriate font mixing (e.g., Arial mixed with Times in official letterhead)
   - Examine if fonts match the expected standards for the document issuer
   - Look for subtle font substitutions that may indicate text replacement

2. **FONT SIZE & SCALING ANALYSIS:**
   - Measure relative font sizes across similar elements
   - Check for unusual size variations in continuous text blocks
   - Verify proportional scaling of related elements (headers vs. subheaders)
   - Identify suspiciously resized text that appears stretched or compressed
   - Examine baseline alignment and consistent line heights

3. **FONT WEIGHT & STYLE INCONSISTENCIES:**
   - Analyze bold, italic, and regular weight distributions
   - Check for artificial bolding or italicization (bitmap effects vs. true font variants)
   - Verify consistent emphasis patterns throughout document
   - Look for weight inconsistencies in similar data fields
   - Examine underline and strikethrough consistency

4. **CHARACTER SPACING & KERNING:**
   - Examine letter spacing (tracking) consistency across text blocks
   - Check for unusual kerning pairs or spacing anomalies
   - Identify artificially adjusted spacing that may hide text modifications
   - Look for inconsistent word spacing patterns
   - Verify character spacing matches font's designed metrics

5. **TEXT RENDERING & QUALITY:**
   - Compare text rendering quality across different sections
   - Check for bitmap vs. vector text inconsistencies
   - Examine anti-aliasing patterns and subpixel rendering
   - Look for text that appears copied/pasted from different sources
   - Identify compression artifacts specific to text regions

6. **BASELINE & ALIGNMENT FORENSICS:**
   - Verify text baseline consistency across lines and paragraphs
   - Check for micro-shifts in text positioning
   - Examine vertical alignment in tabular data
   - Look for text that doesn't follow natural baseline flow
   - Identify signs of manual text positioning vs. natural typesetting

7. **SPECIALIZED TYPOGRAPHY ELEMENTS:**
   - Examine numbers, currency symbols, and special characters
   - Check date formatting and numerical consistency
   - Verify mathematical symbols and formulae rendering
   - Analyze signature fonts vs. body text fonts
   - Review stamp and seal typography authenticity

**DOCUMENT-SPECIFIC FONT STANDARDS:**

**Financial Documents:**
- Banking: Arial, Helvetica for digital, specific bank proprietary fonts
- Insurance: Standard corporate fonts, consistent claim number formatting
- Investment: Regulated disclosure fonts, standardized financial symbol sets

**Government Documents:**
- Official forms: Specific government agency fonts (Times New Roman, Arial common)
- Certificates: Formal serif fonts for main content, sans-serif for details
- Legal documents: Traditional legal fonts, consistent case numbering

**Medical Documents:**
- Hospital letterheads: Institution-specific fonts and formatting
- Prescription forms: Standardized medical fonts for clarity
- Lab reports: Technical fonts for data, consistent unit formatting

**Commercial Documents:**
- Invoices: Business standard fonts, consistent pricing format
- Receipts: Point-of-sale fonts, thermal printer characteristics
- Contracts: Legal document fonts, standardized clause formatting

**FORGERY DETECTION TECHNIQUES:**

1. **Macro Analysis:**
   - Overall document font ecosystem evaluation
   - Institutional font standard compliance
   - Historical font usage patterns for document type

2. **Micro Analysis:**
   - Individual character examination for consistency
   - Pixel-level font rendering comparison
   - Subpixel analysis for copy-paste detection

3. **Contextual Analysis:**
   - Font appropriateness for document date and origin
   - Technology consistency (computer fonts vs. typewriter)
   - Regional font standards and practices

**RED FLAGS FOR FONT-BASED FORGERY:**
- Multiple font families in single-purpose fields
- Inconsistent number formatting in financial data
- Mixed rendering quality suggesting multiple sources
- Anachronistic fonts (modern fonts in old documents)
- Institutional font violations (wrong fonts for known organizations)
- Micro-alignment issues suggesting manual text placement
- Inconsistent character spacing in critical fields
- Quality variations suggesting OCR-generated text insertion

**OUTPUT REQUIREMENTS:**
1. **Overall Assessment:** 'FAKE' or 'LEGIT' determination
2. **Font Profile:** Complete inventory of fonts detected
3. **Inconsistency Map:** Specific locations and types of font problems
4. **Severity Rating:** Critical/Major/Minor inconsistencies identified
5. **Institutional Compliance:** Adherence to expected font standards

**CRITICAL ANALYSIS INSTRUCTION:**
If marking as FAKE, provide:
- Exact font inconsistencies with precise location descriptions
- Technical typography analysis (spacing, rendering, quality differences)
- Specific text examples showing problematic font usage
- Comparison to expected font standards for this document type
- Character-level analysis of suspicious text modifications

Current date: {self.current_date}

Conduct a systematic font forensics analysis with technical precision."""
        
        return self._analyze_with_prompt(file_path, analysis_prompt, "font_inconsistencies", number_of_pages)

    def date_verification_agent(self, file_path: str, number_of_pages: int = 1) -> Dict:
        """
        Analyze document for date verification issues
        
        Args:
            file_path: Path to the document file
            number_of_pages: Number of pages to analyze (for PDFs)
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Running date verification agent")
        
        analysis_prompt = f"""You are a temporal forensics expert specializing in chronological analysis and date verification for document authentication. Perform a comprehensive temporal analysis to detect impossible, inconsistent, or manipulated dates.

**COMPREHENSIVE DATE VERIFICATION FRAMEWORK:**

1. **CHRONOLOGICAL LOGIC ANALYSIS:**
   - **Service/Event Sequence:** Verify logical progression of dates
     • Service dates must precede billing dates
     • Issue dates must precede due dates  
     • Authorization dates must precede service delivery
     • Prescription dates must precede dispensing dates
   - **Business Logic Validation:** 
     • Payment terms compliance (30-day net, etc.)
     • Legal filing deadlines adherence
     • Insurance claim timelines
     • Medical appointment scheduling logic

2. **DATE FORMAT CONSISTENCY:**
   - **Format Standardization:** Check for consistent date formats within document type
     • MM/DD/YYYY vs. DD/MM/YYYY vs. YYYY-MM-DD
     • Month abbreviations (Jan, January, 01) consistency
     • Day of week notation accuracy
   - **Institutional Standards:** Verify format matches issuing organization
     • Government agencies: Standard federal formats
     • Medical institutions: Healthcare industry standards
     • Financial institutions: Banking date conventions
     • International documents: ISO 8601 compliance

3. **TEMPORAL PLAUSIBILITY ASSESSMENT:**
   - **Future Date Detection:** Identify impossible future dates
     • Current date reference: {self.current_date}
     • Service dates in future (unless appointments)
     • Historical document future-dating
   - **Historical Accuracy:** Verify dates align with known timelines
     • Business establishment dates
     • Regulatory implementation dates
     • Technology availability dates
     • Document template introduction dates

4. **WEEKDAY-DATE CORRELATION:**
   - **Calendar Verification:** Cross-check stated days with actual calendar dates
     • "Monday, March 15, 2024" accuracy verification
     • Holiday date accuracy (known federal/state holidays)
     • Business day logic (no services on weekends for business docs)
   - **Pattern Recognition:** Identify systematic weekday errors suggesting automation

5. **DOCUMENT LIFECYCLE ANALYSIS:**
   - **Creation-to-Processing Timeline:**
     • Invoice creation to payment processing duration
     • Medical service to billing statement timing
     • Legal filing to court processing intervals
   - **Revision Dating:** Check for proper version control dating
     • Amendment dates relative to original
     • Correction notice timing
     • Approval workflow chronology

6. **CONTEXT-SPECIFIC DATE VALIDATION:**

   **MEDICAL DOCUMENTS:**
   - Appointment scheduling: Realistic booking timelines
   - Treatment sequences: Logical medical progression
   - Prescription validity: Standard duration limits
   - Insurance authorization: Pre-approval requirements
   - Lab results: Processing time standards

   **FINANCIAL DOCUMENTS:**
   - Banking: Processing day standards, business day rules
   - Insurance: Claim filing deadlines, policy effective dates
   - Investment: Trading day restrictions, settlement periods
   - Credit: Billing cycle consistency, payment due dates

   **LEGAL DOCUMENTS:**
   - Court filings: Jurisdiction-specific deadlines
   - Contracts: Execution and effective date relationships
   - Notarization: Same-day or reasonable proximity requirements
   - Appeals: Statutory timeline compliance

   **GOVERNMENT DOCUMENTS:**
   - Licensing: Application and approval timelines
   - Permits: Processing duration standards
   - Citations: Issue and hearing date relationships
   - Benefits: Eligibility and effective date logic

7. **DIGITAL TIMESTAMP FORENSICS:**
   - **Metadata Analysis:** Compare visible dates with creation timestamps
   - **Version Control:** Check for backdated modifications
   - **Printing Dates:** Verify print dates align with document dates
   - **Electronic Signatures:** Timestamp accuracy for e-signatures

8. **GEOGRAPHIC TIME ZONE CONSIDERATIONS:**
   - **Multi-jurisdiction Documents:** Time zone consistency
   - **International Transactions:** Date line considerations
   - **Regional Date Conventions:** Country-specific formatting
   - **Daylight Saving Time:** Accurate time references

**SOPHISTICATED FORGERY DETECTION:**

1. **Statistical Analysis:**
   - Date distribution patterns (clustering, gaps)
   - Business day vs. weekend date ratios
   - Holiday exclusion patterns
   - Seasonal appropriateness

2. **Cross-Document Correlation:**
   - Referenced document date consistency
   - Multi-page date progression logic
   - Related transaction chronology
   - Supporting document alignment

3. **Institutional Pattern Matching:**
   - Organization-specific date practices
   - Billing cycle adherence
   - Processing timeline standards
   - Historical pattern consistency

**CRITICAL RED FLAGS:**
- Future dates for completed services
- Impossible date combinations (February 30th)
- Inconsistent date formats within single institution
- Weekday-date mismatches on official documents
- Violation of known business processing timelines
- Dates predating organizational establishment
- Inconsistent time zone implications
- Retroactive dating beyond reasonable correction periods
- Holiday processing on non-business days
- Sequence violations (payment before invoice)

**ANALYSIS METHODOLOGY:**
1. **Extract All Dates:** Identify every temporal reference
2. **Format Analysis:** Catalog all date presentation styles
3. **Sequence Mapping:** Plot chronological relationships
4. **Logic Testing:** Verify business rule compliance
5. **Context Validation:** Check institutional standard adherence
6. **Anomaly Detection:** Flag statistical outliers

**OUTPUT REQUIREMENTS:**
1. **Overall Assessment:** 'FAKE' or 'LEGIT' determination
2. **Date Inventory:** Complete catalog of all dates found
3. **Chronological Map:** Timeline of events with logic verification
4. **Violation Details:** Specific temporal inconsistencies identified
5. **Risk Assessment:** Severity ranking of date-related issues

**CRITICAL ANALYSIS MANDATE:**
If marking as FAKE, provide:
- Specific date inconsistencies with exact quotes from document
- Chronological logic violations with detailed explanations
- Format inconsistencies with examples from the document
- Business rule violations specific to document type
- Calendar accuracy issues with precise date references
- Institutional standard deviations with supporting evidence

Current reference date: {self.current_date}
Conduct systematic temporal forensics with mathematical precision."""
        
        return self._analyze_with_prompt(file_path, analysis_prompt, "date_verification", number_of_pages)

    def layout_issues_agent(self, file_path: str, number_of_pages: int = 1) -> Dict:
        """
        Analyze document for layout and alignment issues
        
        Args:
            file_path: Path to the document file
            number_of_pages: Number of pages to analyze (for PDFs)
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Running layout issues agent")
        
        analysis_prompt = f"""You are a document layout forensics expert specializing in detecting structural inconsistencies and formatting anomalies that indicate document tampering or forgery. Perform a comprehensive spatial and structural analysis.

**COMPREHENSIVE LAYOUT ANALYSIS FRAMEWORK:**

1. **SPATIAL STRUCTURE ANALYSIS:**
   - **Margin Consistency:** Verify uniform margins throughout document
     • Top, bottom, left, right margin measurements
     • Consistent margin ratios across pages
     • Standard margin sizes for document type
     • Margin violations suggesting content insertion
   - **Grid Alignment:** Check adherence to invisible layout grid
     • Text block alignment with baseline grid
     • Column structure consistency
     • Tabular data alignment patterns
     • Header/footer positioning accuracy

2. **TEXT BLOCK FORMATTING:**
   - **Paragraph Structure:** Analyze paragraph spacing and indentation
     • Consistent first-line indentation
     • Uniform paragraph spacing (before/after)
     • Justified text alignment accuracy
     • Line height consistency within blocks
   - **Text Flow Analysis:** Verify natural text flow patterns
     • Reading order logic (left-to-right, top-to-bottom)
     • Text wrapping behavior around images/objects
     • Column break positioning
     • Page break logic and widow/orphan control

3. **TABULAR DATA FORENSICS:**
   - **Table Structure Integrity:**
     • Column width consistency
     • Row height uniformity
     • Cell padding standardization
     • Border thickness and style consistency
   - **Data Alignment Patterns:**
     • Numerical right-alignment standards
     • Text left-alignment conventions
     • Decimal point alignment in financial data
     • Header alignment with data columns

4. **VISUAL ELEMENT POSITIONING:**
   - **Logo and Letterhead Placement:**
     • Standard positioning for organizational branding
     • Proportional sizing relative to document
     • Consistent placement across multi-page documents
     • Proper spacing from text content
   - **Signature Block Analysis:**
     • Standard signature line positioning
     • Date field alignment with signatures
     • Title and name block consistency
     • Appropriate spacing from document body

5. **STAMPING AND MARKING FORENSICS:**
   - **Official Stamp Positioning:**
     • Typical placement patterns for document type
     • Overlap behavior with existing content
     • Rotation angles and positioning logic
     • Multiple stamp interaction patterns
   - **Annotation and Mark Analysis:**
     • Handwritten note positioning
     • Highlighting and markup consistency
     • Date stamp placement patterns
     • Correction mark positioning logic

6. **DOCUMENT TYPE SPECIFIC STANDARDS:**

   **FINANCIAL DOCUMENTS:**
   - **Banking Statements:**
     • Standard header layout (bank logo, account info)
     • Transaction table formatting standards
     • Balance positioning conventions
     • Footer information placement
   - **Invoices and Receipts:**
     • Header information hierarchy
     • Line item table structure
     • Total calculation section layout
     • Payment terms positioning

   **MEDICAL DOCUMENTS:**
   - **Prescription Forms:**
     • DEA number and physician info placement
     • Medication table structure
     • Signature block positioning
     • Security feature integration
   - **Medical Records:**
     • Patient header information layout
     • Chart formatting standards
     • Provider signature positioning
     • Date/time stamp placement

   **LEGAL DOCUMENTS:**
   - **Court Filings:**
     • Caption formatting requirements
     • Page numbering standards
     • Signature block requirements
     • Certificate of service layout
   - **Contracts:**
     • Clause numbering and indentation
     • Signature page layout
     • Initial placement conventions
     • Amendment formatting standards

   **GOVERNMENT DOCUMENTS:**
   - **Official Forms:**
     • Field label alignment
     • Checkbox and input field spacing
     • Government seal placement
     • Form number positioning

7. **ADVANCED LAYOUT DETECTION:**
   - **Template Compliance Analysis:**
     • Adherence to known official templates
     • Version-specific layout requirements
     • Institutional formatting standards
     • Regulatory compliance formatting
   - **Professional Typesetting Indicators:**
     • Proper use of typographic conventions
     • Consistent style sheet application
     • Professional spacing relationships
     • Quality typesetting vs. amateur formatting

8. **GEOMETRIC CONSISTENCY:**
   - **Angular Measurements:**
     • Text baseline angles
     • Logo rotation consistency
     • Table border alignment
     • Scan/photo perspective consistency
   - **Proportional Relationships:**
     • Element size relationships
     • Scaling consistency across similar elements
     • Aspect ratio maintenance
     • Spatial hierarchy logic

**SOPHISTICATED FORGERY DETECTION:**

1. **Copy-Paste Detection:**
   - Identical spacing patterns suggesting cloned content
   - Unnatural alignment of inserted elements
   - Background pattern disruption
   - Inconsistent element integration

2. **Digital Manipulation Signatures:**
   - Manual positioning artifacts
   - Grid snapping evidence
   - Layer composition inconsistencies
   - Resolution mismatches between elements

3. **Template Violations:**
   - Deviations from known official layouts
   - Anachronistic formatting for document date
   - Institutional standard violations
   - Version inconsistencies

**CRITICAL RED FLAGS:**
- Micro-alignment issues suggesting manual positioning
- Inconsistent spacing patterns within similar elements
- Elements that violate established geometric relationships
- Text blocks that don't follow natural flow patterns
- Tables with inconsistent internal structure
- Stamps or signatures positioned illogically
- Margin violations without clear justification
- Background pattern disruptions around modified content
- Multi-page formatting inconsistencies
- Professional formatting mixed with amateur corrections

**MEASUREMENT ANALYSIS:**
- **Precise Spacing Analysis:** Measure pixel-level spacing consistency
- **Alignment Grid Detection:** Identify underlying grid structures
- **Proportional Analysis:** Calculate mathematical relationships
- **Statistical Consistency:** Analyze pattern deviation significance

**OUTPUT REQUIREMENTS:**
1. **Overall Assessment:** 'FAKE' or 'LEGIT' determination
2. **Layout Profile:** Complete structural analysis summary
3. **Spatial Violations:** Specific positioning and alignment issues
4. **Template Compliance:** Adherence to expected standards
5. **Geometric Analysis:** Mathematical relationship verification

**CRITICAL ANALYSIS REQUIREMENTS:**
If marking as FAKE, provide:
- Specific layout inconsistencies with precise location descriptions
- Spatial analysis showing alignment or spacing violations
- Geometric measurements demonstrating structural problems
- Template standard deviations with supporting evidence
- Comparative analysis to expected formatting norms
- Technical measurements supporting layout violation claims

Current date: {self.current_date}
Conduct precise geometric and spatial forensics analysis."""
        
        return self._analyze_with_prompt(file_path, analysis_prompt, "layout_issues", number_of_pages)

    def design_verification_agent(self, file_path: str, number_of_pages: int = 1) -> Dict:
        """
        Analyze document for design and branding inconsistencies
        
        Args:
            file_path: Path to the document file
            number_of_pages: Number of pages to analyze (for PDFs)
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Running design verification agent")
        
        analysis_prompt = f"""You are a visual design forensics expert and brand authentication specialist. Perform comprehensive analysis of visual design elements, branding consistency, and security features to detect document forgeries and unauthorized reproductions.

**COMPREHENSIVE DESIGN VERIFICATION FRAMEWORK:**

1. **LOGO AND BRANDING AUTHENTICATION:**
   - **Logo Integrity Analysis:**
     • Pixel-perfect logo reproduction verification
     • Color accuracy and gradient fidelity
     • Proportional scaling and aspect ratio maintenance
     • Vector vs. bitmap quality assessment
     • Logo resolution appropriate for document type
   - **Brand Guidelines Compliance:**
     • Official color palette adherence (Pantone, CMYK, RGB values)
     • Typography consistency with brand standards
     • Logo placement and clear space requirements
     • Minimum size requirements compliance
     • Co-branding and subsidiary logo rules

2. **COLOR ANALYSIS AND VERIFICATION:**
   - **Color Accuracy Assessment:**
     • Spot color reproduction fidelity
     • Corporate color standard compliance
     • Color temperature and saturation consistency
     • Print vs. digital color representation
     • Color profile accuracy for document type
   - **Color Consistency Patterns:**
     • Uniform color application across elements
     • Color bleeding or registration issues
     • Ink density variations suggesting reproduction
     • Color gamut limitations indicating forgery source

3. **SECURITY FEATURE VERIFICATION:**
   - **Physical Security Elements:**
     • Watermark presence and authenticity
     • Security threads and embedded features
     • Microprinting visibility and clarity
     • Holographic elements and iridescent features
     • UV-reactive inks and hidden elements
   - **Digital Security Features:**
     • Digital watermarks and steganography
     • QR codes and 2D barcode verification
     • Security fonts and specialized typefaces
     • Copy-evident backgrounds and void pantographs
     • Anti-scan/anti-copy patterns

4. **INSTITUTIONAL DESIGN STANDARDS:**

   **GOVERNMENT DOCUMENTS:**
   - **Federal Agencies:**
     • Official seal specifications and placement
     • Government typography standards (typically Arial, Times New Roman)
     • Standard color schemes (federal blue, official red)
     • Security feature requirements by document type
     • Anti-counterfeiting measures verification
   - **State and Local:**
     • Jurisdiction-specific design requirements
     • State seal and emblem accuracy
     • Local government branding standards
     • Official form layout requirements

   **FINANCIAL INSTITUTIONS:**
   - **Banking Design Standards:**
     • Bank logo authentication and brand compliance
     • Routing number and account formatting
     • Security features (MICR encoding, special papers)
     • Statement layout and design consistency
     • Anti-fraud design elements
   - **Investment and Insurance:**
     • Regulatory compliance design requirements
     • Industry-standard security features
     • Professional presentation standards
     • Disclosure formatting requirements

   **HEALTHCARE ORGANIZATIONS:**
   - **Hospital and Clinic Branding:**
     • Medical institution logo verification
     • Healthcare typography standards
     • HIPAA compliance design elements
     • Medical form layout standards
     • Patient information security features
   - **Pharmaceutical:**
     • Drug manufacturer branding
     • Prescription form security features
     • DEA and FDA compliance elements
     • Anti-counterfeiting measures

   **EDUCATIONAL INSTITUTIONS:**
   - **University and School Design:**
     • Institutional seal and logo accuracy
     • Academic formatting standards
     • Transcript security features
     • Certificate and diploma design verification
     • Official letterhead standards

5. **ADVANCED DESIGN FORENSICS:**
   - **Print Technology Analysis:**
     • Offset printing characteristics identification
     • Digital printing artifact detection
     • Laser printer toner analysis
     • Inkjet printing pattern recognition
     • Thermal printing characteristic verification
   - **Production Quality Assessment:**
     • Professional vs. consumer-grade reproduction
     • Print resolution and dot pattern analysis
     • Paper stock and texture verification
     • Binding and finishing quality assessment
     • Age-appropriate production techniques

6. **VISUAL DESIGN CONSISTENCY:**
   - **Design System Coherence:**
     • Consistent visual hierarchy implementation
     • Style guide adherence across elements
     • Grid system compliance
     • Visual weight and balance assessment
     • Information architecture standards
   - **Professional Design Indicators:**
     • Quality of graphic design execution
     • Appropriate use of design principles
     • Professional typography treatment
     • Consistent visual language application
     • Brand identity system integration

7. **HISTORICAL AND CONTEXTUAL VERIFICATION:**
   - **Temporal Design Accuracy:**
     • Logo version appropriate for document date
     • Design trends and styles period-appropriate
     • Technology limitations for claimed creation date
     • Brand evolution timeline compliance
     • Historical accuracy of visual elements
   - **Geographic and Cultural Appropriateness:**
     • Region-specific design standards
     • Cultural considerations in visual design
     • Local vs. international branding variations
     • Language-specific typography requirements

8. **DIGITAL REPRODUCTION ANALYSIS:**
   - **Scan and Photography Assessment:**
     • Natural lighting and shadow patterns
     • Scanner bed artifacts and limitations
     • Photography perspective and distortion
     • Digital compression impact on design elements
     • Screen moire patterns and digital interference
   - **Digital Manipulation Detection:**
     • Layer composition analysis
     • Resolution inconsistencies between elements
     • Digital editing artifacts in design elements
     • Unnatural perfection suggesting digital recreation
     • Composite image detection techniques

**SOPHISTICATED FORGERY INDICATORS:**

1. **Amateur Design Execution:**
   - Inconsistent visual hierarchy
   - Poor typography choices and spacing
   - Color reproduction inaccuracies
   - Low-resolution logo reproduction
   - Unprofessional design composition

2. **Brand Standard Violations:**
   - Incorrect logo versions or modifications
   - Unauthorized color variations
   - Improper logo placement or sizing
   - Missing required design elements
   - Non-compliant brand usage

3. **Security Feature Anomalies:**
   - Missing expected security elements
   - Poorly reproduced security features
   - Inconsistent security feature integration
   - Age-inappropriate security technology
   - Obvious security feature simulation

**CRITICAL RED FLAGS:**
- Logo distortion or pixelation indicating low-quality source
- Color inaccuracies suggesting non-official reproduction
- Missing institutional security features
- Anachronistic design elements for document date
- Design quality inconsistent with organizational standards
- Security features that appear simulated or reproduced
- Brand guideline violations indicating unauthorized creation
- Design composition suggesting amateur execution
- Visual elements inappropriate for claimed institution
- Missing required regulatory or compliance design elements

**VERIFICATION METHODOLOGY:**
1. **Element Cataloging:** Identify all visual design elements
2. **Brand Standard Comparison:** Compare against known official standards
3. **Security Feature Analysis:** Verify presence and authenticity of security elements
4. **Quality Assessment:** Evaluate professional design execution
5. **Historical Verification:** Confirm period-appropriate design elements
6. **Technical Analysis:** Examine production and reproduction characteristics

**OUTPUT REQUIREMENTS:**
1. **Overall Assessment:** 'FAKE' or 'LEGIT' determination
2. **Design Authentication:** Brand compliance and logo verification
3. **Security Feature Report:** Presence and authenticity of security elements
4. **Quality Analysis:** Professional design execution assessment
5. **Violation Catalog:** Specific design and brand standard deviations

**CRITICAL ANALYSIS MANDATE:**
If marking as FAKE, provide:
- Specific design inconsistencies with exact visual element descriptions
- Brand standard violations with detailed explanations
- Security feature analysis showing missing or reproduced elements
- Logo and visual identity authentication results
- Production quality assessment indicating forgery techniques
- Historical accuracy evaluation of design elements
- Technical analysis of reproduction methods used

Current date: {self.current_date}
Conduct systematic visual forensics and brand authentication analysis."""
        
        return self._analyze_with_prompt(file_path, analysis_prompt, "design_verification", number_of_pages)

    def content_analysis_agent(self, file_path: str, number_of_pages: int = 1) -> Dict:
        """
        Analyze document for content and logical inconsistencies
        
        Args:
            file_path: Path to the document file
            number_of_pages: Number of pages to analyze (for PDFs)
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Running content analysis agent")
        
        analysis_prompt = f"""You are a content authenticity expert and logical consistency analyst specializing in detecting document forgeries through comprehensive content verification. Perform systematic analysis of document information, calculations, and logical relationships.

**COMPREHENSIVE CONTENT ANALYSIS FRAMEWORK:**

1. **MATHEMATICAL ACCURACY VERIFICATION:**
   - **Financial Calculations:**
     • Invoice subtotals, taxes, and total calculations
     • Discount applications and percentage calculations
     • Interest rate calculations and compound interest
     • Currency conversions and exchange rate accuracy
     • Payment schedules and amortization calculations
   - **Statistical and Data Analysis:**
     • Percentage calculations and ratio accuracy
     • Statistical summaries and data aggregations
     • Measurement conversions and unit consistency
     • Scientific calculations and formulae verification
     • Data distribution patterns and logical ranges

2. **INTERNAL CONSISTENCY VERIFICATION:**
   - **Cross-Reference Analysis:**
     • Account numbers, reference numbers, and ID consistency
     • Name and address information across document sections
     • Date references and timeline consistency
     • Quantity and measurement consistency throughout
     • Contact information and communication details accuracy
   - **Data Relationship Logic:**
     • Parent-child data relationships (header to line items)
     • Hierarchical information consistency
     • Related field interdependencies
     • Cascade effect verification (changes propagating correctly)
     • Business rule compliance verification

3. **DOMAIN-SPECIFIC CONTENT VALIDATION:**

   **FINANCIAL DOCUMENTS:**
   - **Banking and Investment:**
     • Account balance reconciliation
     • Transaction sequence logic
     • Interest calculation accuracy
     • Fee structure compliance
     • Regulatory disclosure completeness
   - **Insurance Documents:**
     • Premium calculation accuracy
     • Coverage limit consistency
     • Deductible application logic
     • Policy term calculation
     • Risk assessment appropriateness
   - **Tax Documents:**
     • Tax calculation accuracy
     • Deduction legitimacy and limits
     • Income reporting consistency
     • Filing status appropriateness
     • Tax code compliance

   **MEDICAL DOCUMENTS:**
   - **Clinical Records:**
     • Diagnosis code (ICD-10) accuracy and consistency
     • Treatment timeline logical progression
     • Medication dosage and frequency appropriateness
     • Laboratory value normal ranges
     • Medical procedure coding accuracy (CPT codes)
   - **Prescription Documents:**
     • Drug interaction checking
     • Dosage appropriateness for patient demographics
     • Prescription duration logical limits
     • Controlled substance regulations compliance
     • Pharmacy and prescriber information accuracy

   **LEGAL DOCUMENTS:**
   - **Court Documents:**
     • Case number and jurisdiction consistency
     • Legal citation accuracy and format
     • Procedural requirement compliance
     • Statutory reference verification
     • Filing deadline calculation accuracy
   - **Contracts and Agreements:**
     • Terms and conditions consistency
     • Legal language appropriateness
     • Execution and effective date logic
     • Party information accuracy
     • Consideration and obligation balance

   **GOVERNMENT DOCUMENTS:**
   - **Official Forms and Permits:**
     • Regulation compliance verification
     • Application requirement fulfillment
     • Fee calculation accuracy
     • Processing timeline appropriateness
     • Authority jurisdiction verification

4. **LINGUISTIC AND CONTEXTUAL ANALYSIS:**
   - **Language and Terminology:**
     • Professional terminology appropriate for document type
     • Industry-specific jargon and acronym usage
     • Regional language variations and spelling conventions
     • Technical term accuracy and context
     • Legal or medical language precision
   - **Communication Style Consistency:**
     • Formal vs. informal tone appropriateness
     • Business communication standards
     • Institutional voice and style consistency
     • Document purpose-appropriate language
     • Professional writing quality assessment

5. **BUSINESS LOGIC VERIFICATION:**
   - **Industry Standard Compliance:**
     • Business process adherence
     • Industry regulation compliance
     • Standard operating procedure consistency
     • Professional practice standards
     • Ethical and legal requirement fulfillment
   - **Operational Reality Checks:**
     • Business hour consistency
     • Service availability and limitations
     • Geographic service area accuracy
     • Resource allocation realism
     • Timeline feasibility assessment

6. **DATA INTEGRITY ANALYSIS:**
   - **Information Completeness:**
     • Required field population
     • Mandatory information presence
     • Supporting documentation references
     • Contact information completeness
     • Legal requirement fulfillment
   - **Data Quality Assessment:**
     • Information accuracy and precision
     • Data format consistency
     • Value range appropriateness
     • Outlier detection and validation
     • Missing or incomplete information identification

7. **REGULATORY AND COMPLIANCE VERIFICATION:**
   - **Legal Requirement Compliance:**
     • Statutory disclosure requirements
     • Regulatory filing standards
     • Industry-specific compliance mandates
     • Consumer protection requirements
     • Privacy and confidentiality standards
   - **Professional Standard Adherence:**
     • Professional licensing requirements
     • Certification and accreditation standards
     • Ethical guideline compliance
     • Industry best practice adherence
     • Quality assurance standards

8. **ADVANCED CONTENT FORENSICS:**
   - **Pattern Recognition:**
     • Unusual data patterns or anomalies
     • Statistical outliers requiring explanation
     • Systematic errors or biases
     • Artificial data generation indicators
     • Template-based content variations
   - **Behavioral Analysis:**
     • Human vs. automated content generation
     • Natural language vs. artificial text patterns
     • Decision-making logic appropriateness
     • Workflow adherence and process compliance
     • User behavior pattern consistency

**SOPHISTICATED FORGERY DETECTION:**

1. **Mathematical Inconsistencies:**
   - Calculation errors in financial documents
   - Statistical impossibilities or implausibilities
   - Rounding errors suggesting manual manipulation
   - Currency or unit conversion mistakes
   - Percentage calculations that don't add up

2. **Logic Violations:**
   - Business rule violations
   - Procedural sequence errors
   - Regulatory compliance failures
   - Industry standard deviations
   - Impossible or improbable scenarios

3. **Content Quality Indicators:**
   - Amateur vs. professional content creation
   - Template filling vs. authentic document generation
   - Copy-paste content assembly
   - Information source inconsistencies
   - Quality variations suggesting multiple authors

**CRITICAL RED FLAGS:**
- Mathematical errors in routine calculations
- Business logic violations for document type
- Industry terminology misuse or errors
- Regulatory compliance failures
- Impossible data combinations or relationships
- Anachronistic content for document date
- Information inconsistencies across document sections
- Professional standard violations
- Language quality inappropriate for claimed source
- Data patterns suggesting artificial generation

**ANALYSIS METHODOLOGY:**
1. **Content Extraction:** Identify all factual claims and data
2. **Mathematical Verification:** Check all calculations and formulas
3. **Logic Testing:** Verify business rules and logical consistency
4. **Cross-Reference Validation:** Check internal consistency
5. **Domain Expertise Application:** Apply industry-specific knowledge
6. **Pattern Analysis:** Identify anomalies and irregularities

**OUTPUT REQUIREMENTS:**
1. **Overall Assessment:** 'FAKE' or 'LEGIT' determination
2. **Content Accuracy Report:** Mathematical and factual verification
3. **Logic Consistency Analysis:** Business rule and logical relationship verification
4. **Domain Compliance Assessment:** Industry and regulatory standard adherence
5. **Quality Evaluation:** Professional content creation assessment

**CRITICAL ANALYSIS REQUIREMENTS:**
If marking as FAKE, provide:
- Specific mathematical errors with exact calculations shown
- Logical inconsistencies with detailed explanations
- Business rule violations with industry standard references
- Content quality issues with supporting evidence
- Cross-reference failures with specific examples
- Domain expertise violations with professional standard citations
- Regulatory compliance failures with specific requirement references

Current date: {self.current_date}
Conduct comprehensive content authenticity and logical consistency analysis."""
        
        return self._analyze_with_prompt(file_path, analysis_prompt, "content_analysis", number_of_pages)

    def _analyze_with_prompt(self, file_path: str, analysis_prompt: str, 
                           indicator_type: str, number_of_pages: int = 1) -> Dict:
        """
        Perform analysis using the specified prompt
        
        Args:
            file_path: Path to the document file
            analysis_prompt: The analysis prompt to use
            indicator_type: Type of analysis being performed
            number_of_pages: Number of pages to analyze
            
        Returns:
            Dictionary with analysis results
        """
        try:
            if file_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")):
                return self._analyze_image(file_path, analysis_prompt, indicator_type)
            elif file_path.lower().endswith(".pdf"):
                return self._analyze_pdf(file_path, analysis_prompt, indicator_type, number_of_pages)
            else:
                return {
                    "module": f"individual_agent_{indicator_type}",
                    "score": 0.0,
                    "suspicious": False,
                    "indicators": [f"Unsupported file format for {indicator_type} analysis"],
                    "reasoning": f"Unsupported file format for {indicator_type} analysis",
                    "details": {}
                }
        except Exception as e:
            logger.error(f"Error in {indicator_type} analysis: {str(e)}")
            return {
                "module": f"individual_agent_{indicator_type}",
                "score": 0.0,
                "suspicious": False,
                "indicators": [f"Error during {indicator_type} analysis: {str(e)}"],
                "reasoning": f"Error during {indicator_type} analysis: {str(e)}",
                "details": {"error": str(e)}
            }

    def _analyze_image(self, image_path: str, prompt: str, indicator_type: str) -> Dict:
        """Analyze a single image"""
        try:
            image = Image.open(image_path)
            
            # Convert image to base64 for API transmission
            buffer = io.BytesIO()
            image.save(buffer, format=image.format or "JPEG")
            base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            if self.openai_client:
                # Use OpenAI
                response = self.openai_client.chat.completions.create(
                    model=self.current_model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are analyzing various types of documents to detect potential forgeries. These could include financial documents, medical bills, invoices, receipts, or any other official documents. Your main task is to check if a document is fake based on the inconsistencies found. The current date is {self.current_date}."
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/{image.format.lower() if image.format else 'jpeg'};base64,{base64_image}"}}
                            ]
                        }
                    ],
                    max_completion_tokens=1000,
                    temperature=self.temperature
                )
                
                reasoning = response.choices[0].message.content
                suspicious = "fake" in reasoning.lower() and "legit" not in reasoning.lower()
                
            else:
                raise Exception("No API available for analysis")
            
            # Calculate score based on suspicion level
            score = 0.8 if suspicious else 0.1
            
            # Extract indicators from reasoning
            indicators = self._extract_indicators_from_reasoning(reasoning, indicator_type)
            
            return {
                "module": f"individual_agent_{indicator_type}",
                "score": score,
                "suspicious": suspicious,
                "indicators": indicators,
                "reasoning": reasoning,
                "details": {
                    "agent_type": indicator_type,
                    "api_used": "OpenAI",
                    "image_analyzed": True
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {str(e)}")
            return {
                "module": f"individual_agent_{indicator_type}",
                "score": 0.0,
                "suspicious": False,
                "indicators": [f"Error during {indicator_type} analysis: {str(e)}"],
                "reasoning": f"Error during {indicator_type} analysis: {str(e)}",
                "details": {"error": str(e), "agent_type": indicator_type}
            }

    def _analyze_pdf(self, pdf_path: str, prompt: str, indicator_type: str, number_of_pages: int) -> Dict:
        """Analyze a PDF document"""
        try:
            pdf_images = convert_from_path(pdf_path, dpi=300)
            pages_to_process = pdf_images[:number_of_pages]

            if not pages_to_process:
                return {
                    "module": f"individual_agent_{indicator_type}",
                    "score": 0.0,
                    "suspicious": False,
                    "indicators": [f"No pages extracted from PDF for {indicator_type} analysis"],
                    "reasoning": f"No pages extracted from PDF for {indicator_type} analysis",
                    "details": {"agent_type": indicator_type}
                }

            # Prepare message content for multi-page analysis
            message_content = [{"type": "text", "text": prompt}]
            
            # Add each page as an image
            for i, image in enumerate(pages_to_process):
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
                message_content.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                })
                logger.debug(f"Added page {i+1} to {indicator_type} analysis")

            message_content.append({
                "type": "text", 
                "text": "Analyze the provided document pages based on the indicators listed previously. Determine if the document is 'fake' or 'legit'."
            })

            if self.openai_client:
                # Use OpenAI
                response = self.openai_client.chat.completions.create(
                    model=self.current_model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are analyzing various types of documents to detect potential forgeries. These could include financial documents, medical bills, invoices, receipts, or any other official documents. Your main task is to check if a document is fake based on the inconsistencies found. The current date is {self.current_date}."
                        },
                        {
                            "role": "user",
                            "content": message_content
                        }
                    ],
                    max_completion_tokens=1500,
                    temperature=self.temperature
                )
                
                reasoning = response.choices[0].message.content
                suspicious = "fake" in reasoning.lower() and "legit" not in reasoning.lower()
                
            else:
                raise Exception("No API available for analysis")

            # Calculate score based on suspicion level
            score = 0.8 if suspicious else 0.1
            
            # Extract indicators from reasoning
            indicators = self._extract_indicators_from_reasoning(reasoning, indicator_type)

            return {
                "module": f"individual_agent_{indicator_type}",
                "score": score,
                "suspicious": suspicious,
                "indicators": indicators,
                "reasoning": reasoning,
                "details": {
                    "agent_type": indicator_type,
                    "api_used": "OpenAI",
                    "pages_analyzed": len(pages_to_process),
                    "pdf_analyzed": True
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing PDF {pdf_path}: {str(e)}")
            return {
                "module": f"individual_agent_{indicator_type}",
                "score": 0.0,
                "suspicious": False,
                "indicators": [f"Error during {indicator_type} PDF analysis: {str(e)}"],
                "reasoning": f"Error during {indicator_type} PDF analysis: {str(e)}",
                "details": {"error": str(e), "agent_type": indicator_type}
            }

    def _extract_indicators_from_reasoning(self, reasoning: str, indicator_type: str) -> List[str]:
        """Extract specific indicators from the AI reasoning text"""
        indicators = []
        
        # Add the primary indicator type if suspicious
        if "fake" in reasoning.lower():
            indicators.append(f"{indicator_type.replace('_', ' ').title()} detected")
        
        # Extract specific issues mentioned in reasoning
        reasoning_lower = reasoning.lower()
        
        # Common forgery indicators to look for
        forgery_keywords = {
            "digital_artifacts": ["compression", "pixelation", "artifacts", "quality", "resolution", "blurry"],
            "font_inconsistencies": ["font", "typeface", "weight", "size", "style", "typography"],
            "date_verification": ["date", "time", "chronological", "sequence", "format"],
            "layout_issues": ["alignment", "spacing", "margin", "layout", "positioning", "structure"],
            "design_verification": ["logo", "branding", "color", "design", "watermark", "seal"],
            "content_analysis": ["calculation", "mathematical", "logical", "inconsistent", "error"]
        }
        
        if indicator_type in forgery_keywords:
            for keyword in forgery_keywords[indicator_type]:
                if keyword in reasoning_lower:
                    indicators.append(f"Potential {keyword} issue identified")
                    break  # Avoid duplicate indicators
        
        return indicators[:3]  # Limit to top 3 indicators

    def run_all_agents(self, file_path: str, number_of_pages: int = 1) -> Dict[str, Dict]:
        """
        Run all individual analysis agents on a document
        
        Args:
            file_path: Path to the document file
            number_of_pages: Number of pages to analyze (for PDFs)
            
        Returns:
            Dictionary with results from all agents
        """
        logger.info(f"Running all individual agents on {file_path}")
        
        agents = {
            "digital_artifacts": self.digital_artifacts_agent,
            "font_inconsistencies": self.font_inconsistencies_agent,
            "date_verification": self.date_verification_agent,
            "layout_issues": self.layout_issues_agent,
            "design_verification": self.design_verification_agent,
            "content_analysis": self.content_analysis_agent
        }
        
        results = {}
        overall_suspicious = False
        all_indicators = []
        total_score = 0.0
        
        for agent_name, agent_func in agents.items():
            try:
                logger.info(f"Running {agent_name} agent")
                result = agent_func(file_path, number_of_pages)
                results[agent_name] = result
                
                # Aggregate results
                if result.get('suspicious', False):
                    overall_suspicious = True
                all_indicators.extend(result.get('indicators', []))
                total_score += result.get('score', 0.0)
                
                logger.info(f"{agent_name} analysis complete: suspicious={result.get('suspicious', False)}, score={result.get('score', 0.0)}")
            except Exception as e:
                logger.error(f"Error running {agent_name} agent: {str(e)}")
                results[agent_name] = {
                    "module": f"individual_agent_{agent_name}",
                    "score": 0.0,
                    "suspicious": False,
                    "indicators": [f"Error in {agent_name} analysis: {str(e)}"],
                    "reasoning": f"Error in {agent_name} analysis: {str(e)}",
                    "details": {"error": str(e), "agent_type": agent_name}
                }
        
        # Calculate average score
        average_score = total_score / len(agents) if agents else 0.0
        
        # Add aggregate summary
        results["_summary"] = {
            "module": "individual_agents_aggregate",
            "score": average_score,
            "suspicious": overall_suspicious,
            "indicators": list(set(all_indicators)),  # Remove duplicates
            "agents_run": list(agents.keys()),
            "total_agents": len(agents),
            "suspicious_agents": sum(1 for r in results.values() if r.get('suspicious', False))
        }
        
        return results

def test_individual_agents():
    """Test function for individual agents"""
    # Test with standard model
    print("=== Testing Individual Agents with Standard Model ===")
    agents_standard = IndividualAnalysisAgents(model_type="standard")
    logger.info(f"Standard agents initialized with model: {agents_standard.current_model_name}")
    
    # Test with reasoning model
    print("\n=== Testing Individual Agents with Reasoning Model ===")
    agents_reasoning = IndividualAnalysisAgents(model_type="reasoning")
    logger.info(f"Reasoning agents initialized with model: {agents_reasoning.current_model_name}")
    
    # You can test with a real document by providing the path
    test_document = "test_document.pdf"  # You'll need to provide a real document
    
    try:
        print(f"\nTesting with standard model ({agents_standard.current_model_name}):")
        results = agents_standard.run_all_agents(test_document, number_of_pages=1)
        
        print("Individual Agent Results:")
        for agent_name, result in results.items():
            if agent_name == "_summary":
                print(f"\n{agent_name.upper()}:")
                print(f"  Overall Suspicious: {result.get('suspicious', False)}")
                print(f"  Average Score: {result.get('score', 0.0):.3f}")
                print(f"  Agents Run: {result.get('total_agents', 0)}")
                print(f"  Suspicious Agents: {result.get('suspicious_agents', 0)}")
            else:
                print(f"\n{agent_name.upper()}:")
                print(f"  Suspicious: {result.get('suspicious', False)}")
                print(f"  Score: {result.get('score', 0.0):.3f}")
                print(f"  Indicators: {len(result.get('indicators', []))}")
                print(f"  Reasoning: {result.get('reasoning', 'No reasoning')[:200]}...")
            
    except Exception as e:
        print(f"Test failed: {str(e)}")
        print("Note: Make sure to provide a valid document path and set up your OpenAI API key")

if __name__ == "__main__":
    test_individual_agents() 