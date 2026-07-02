"""Capability → allowed-divisions map (the OCL, in code).

Ported verbatim from the permission engine's audited `OVERRIDE_CAP_LOOKUP`
(merged over `INDIVIDUAL_CAPS`). Each capability ID from
`capability_matrix.csv` maps to the set of Algolia divisions allowed to
read tables classified under it. This is the access-defining artifact —
change it via a reviewed PR; it does not depend on usage history and is
computed live per-PR, so there is no `governance_state.yaml` to keep in sync.
"""

from __future__ import annotations

# capability_id -> (title, frozenset(allowed_divisions))
CAPABILITY_DIVISIONS: dict[str, tuple[str, frozenset[str]]] = {
    'SAP': ('Search & Discovery Platform Management', frozenset({'AI Search', 'AI Search | Ranking', 'AI Search | Understanding', 'Data Platform', 'Developer Experience', 'IaaS', 'PaaS', 'R&D Leadership', 'Scale', 'Technical Programs'})),
    'NES': ('NeuralSearch & Hybrid Retrieval', frozenset({'AI', 'AI Research', 'AI Search', 'AI Search | Ranking', 'AI Search | Understanding', 'AI | AI Platform', 'R&D Leadership', 'Relevance', 'Search Core', 'Search NLU'})),
    'SCI': ('Search Configuration & Index Management', frozenset({'AI', 'AI Research', 'AI Search', 'AI Search | Understanding', 'Customer Solutions', 'Developer Experience', 'Product Operations & Development', 'R&D Leadership', 'Scale', 'Solutions Engineers'})),
    'ING': ('Data Indexing & Ingestion', frozenset({'AI Search', 'Data | Connections', 'Data | E-commerce Integrations', 'Data | Magento', 'Events & Records Connection', 'Events & Records Pipeline', 'Integrations', 'R&D Leadership', 'Scale', 'Search Core'})),
    'SME': ('Search / Merchandising', frozenset({'AI Search', 'AI Search | Ranking', 'Application Experiences', 'Control', 'Control | Merchandising', 'Discovery', 'Guided Relevance', 'Product Management', 'R&D Leadership', 'Recommend'})),
    'RCE': ('Recommendations Engine', frozenset({'AI', 'AI Research', 'AI | AI Platform', 'Application Experiences', 'Data | Application Experiences', 'Discovery', 'Product Management', 'R&D Leadership', 'Recommend'})),
    'AIM': ('AI Model Management & MLOps', frozenset({'AI', 'AI Research', 'AI Search', 'AI Search | Personalization', 'AI Search | Ranking', 'AI Search | Understanding', 'AI | AI Platform', 'AI | Optimization', 'R&D Leadership', 'Recommend'})),
    'MLA': ('Machine Learning / Auto-ML', frozenset({'AI', 'AI Research', 'AI Search', 'AI Search | Ranking', 'AI Search | Understanding', 'AI | AI Platform', 'AI | Optimization', 'R&D Leadership', 'Recommend', 'Relevance'})),
    'ADD': ('AI Driven Discovery', frozenset({'AI', 'AI Research', 'AI Search', 'AI | AI Platform', 'Data Analytics', 'Data Platform', 'Discovery', 'Leadership', 'R&D Leadership', 'Recommend'})),
    'NBA': ('Next Best Action / Recommendation', frozenset({'AI', 'AI Research', 'AI Search | Personalization', 'AI Search | Ranking', 'AI | AI Platform', 'Customer Solutions', 'Discovery', 'Product Management', 'R&D Leadership', 'Recommend'})),
    'PAI': ('Predictive Analytics / Intelligence', frozenset({'AI', 'AI Research', 'AI | AI Platform', 'CEO Office', 'Data Analysis', 'Data Analytics', 'Leadership', 'Pricing & Monetization Strategy', 'Revenue Operations'})),
    'NSA': ('NLP & Sentiment Analysis', frozenset({'AI', 'AI Research', 'AI Search | Understanding', 'Customer Care', 'Customer Support', 'Data Analytics', 'Product Management', 'Search NLU'})),
    'DLE': ('Deep Learning', frozenset({'AI', 'AI Research', 'AI Search', 'AI Search | Understanding', 'AI | AI Platform', 'AI | Optimization', 'R&D Leadership', 'Recommend', 'Search NLU'})),
    'AGT': ('Agentic AI Platform', frozenset({'AI', 'AI Research', 'AI Search', 'AI Search | Understanding', 'AI | AI Platform', 'Application Experiences', 'DevRel', 'Developer Experience', 'Product Management', 'R&D Leadership'})),
    'ICH': ('Intelligent Chatbots', frozenset({'AI', 'AI Research', 'AI Search', 'AI Search | Understanding', 'AI | AI Platform', 'Customer Solutions', 'Developer Experience', 'Product Management', 'R&D Leadership', 'Solutions Engineers'})),
    'DXE': ('Developer Experience & SDK Management', frozenset({'DX', 'DelRel', 'Dev Journey', 'DevRel', 'Developer Experience', 'Developer Relations', 'IS Mobile', 'Instant Web', 'Integrations', 'R&D Leadership'})),
    'DRE': ('Developer Relations & Community', frozenset({'CEO Office', 'Community', 'Content', 'DelRel', 'Dev Journey', 'DevRel', 'Developer Experience', 'Developer Relations', 'Events', 'Marketing Leadership'})),
    'AMG': ('API Management / Gateway', frozenset({'Core Api', 'DX', 'Developer Experience', 'Engineering Security', 'Foundation', 'Infrastructure', 'R&D Leadership', 'Search Platform', 'Technical Programs'})),
    'HMS': ('Headless Microservice Support', frozenset({'Core Api', 'DX', 'DevRel', 'Developer Experience', 'Developer Relations', 'Foundation', 'Integrations', 'R&D Leadership', 'Technical Programs'})),
    'MSP': ('Marketing Strategy & Planning', frozenset({'CEO Office', 'Communication', 'Corporate Communications', 'Corporate Marketing', 'Demand Generation', 'Field Marketing', 'Growth Marketing', 'Marketing Leadership', 'Marketing Operations', 'Product Marketing'})),
    'LGM': ('Lead Generation & Management', frozenset({'Demand Generation', 'Digital Marketing', 'Field Marketing', 'Growth', 'Inbound - SDR', 'Marketing Operations', 'Outbound - BDR', 'Revenue Operations', 'Sales Operations'})),
    'GDX': ('Growth & Developer Acquisition', frozenset({'CEO Office', 'Data Analysis', 'Dev Journey', 'Developer Relations', 'Growth', 'Growth Marketing', 'Marketing Leadership', 'Marketing Operations', 'Product Marketing'})),
    'MJM': ('Journey Management', frozenset({'CEO Office', 'Demand Generation', 'Dev Journey', 'Digital Marketing', 'Field Marketing', 'Growth', 'Growth Marketing', 'Marketing Operations', 'Product Marketing', 'Revenue Operations'})),
    'DCP': ('Digital Commerce & Personalisation', frozenset({'CEO Office', 'Corporate Marketing', 'Digital Marketing', 'Growth', 'Growth Marketing', 'Marketing Leadership', 'Marketing Operations', 'Product Marketing', 'Web & Design'})),
    'MM': ('Marketing Measurement', frozenset({'CEO Office', 'Corporate Marketing', 'Data Analysis', 'Demand Generation', 'Digital Marketing', 'Field Marketing', 'Growth', 'Marketing Operations', 'Product Marketing', 'Revenue Operations'})),
    'AMS': ('Asset & Content Management', frozenset({'Content', 'Corporate Communications', 'Corporate Marketing', 'Dev Journey', 'Developer Relations', 'Marketing Leadership', 'Marketing Operations', 'Product Marketing', 'Web & Design'})),
    'MOS': ('Omni-channel & Social', frozenset({'Community', 'Corporate Communications', 'Corporate Marketing', 'DevRel', 'Developer Relations', 'Digital Marketing', 'Marketing Leadership', 'Marketing Operations', 'Product Marketing', 'Web & Design'})),
    'PM3': ('Paid Marketing', frozenset({'CEO Office', 'Demand Generation', 'Digital Marketing', 'Field Marketing', 'Growth', 'Growth Marketing', 'Marketing Leadership', 'Marketing Operations', 'Revenue Operations'})),
    'AOG': ('Audience & Segmentation', frozenset({'CEO Office', 'Data Analysis', 'Demand Generation', 'Digital Marketing', 'Field Marketing', 'Growth', 'Marketing Operations', 'Product Marketing', 'Revenue Operations', 'Sales Operations'})),
    'SO': ('Sales Operations', frozenset({'CEO Office', 'Commercial', 'Enterprise', 'Leadership', 'MM', 'Marketing Operations', 'Revenue Operations', 'SMB', 'Sales Leadership', 'Sales Operations', 'Small Medium Business'})),
    'SRM': ('Sales Relationship Management', frozenset({'CEO Office', 'Commercial', 'Customer Success', 'Enterprise', 'Leadership', 'MM', 'Marketing Operations', 'Mid Market', 'Revenue Operations', 'Sales Leadership', 'Small Medium Business'})),
    'OM1': ('Opportunity Management', frozenset({'CEO Office', 'Commercial', 'Customer Solutions', 'Enterprise', 'Leadership', 'MM', 'Marketing Operations', 'Mid Market', 'Revenue Operations', 'Sales Leadership', 'Sales Operations', 'Small Medium Business'})),
    'CUS': ('Cross / Up-Sell Management', frozenset({'CEO Office', 'Commercial', 'Customer Success', 'Data Analysis', 'Enterprise', 'Leadership', 'MM', 'Revenue Operations', 'Sales Leadership', 'Sales Operations'})),
    'SE': ('Sales Enablement', frozenset({'CEO Office', 'Commercial', 'Enterprise', 'Leadership', 'MM', 'Marketing Operations', 'Mid Market', 'Product Marketing', 'Sales Enablement', 'Sales Leadership', 'Solutions Engineers'})),
    'ASI': ('Analytics, Scoring & Intelligence', frozenset({'CEO Office', 'Commercial', 'Data Analysis', 'Enterprise', 'Leadership', 'MM', 'Revenue Operations', 'Sales Leadership', 'Sales Operations', 'Small Medium Business'})),
    'TMA': ('Territory Management', frozenset({'CEO Office', 'Commercial', 'Enterprise', 'Leadership', 'MM', 'Revenue Operations', 'Sales Leadership', 'Sales Operations', 'Small Medium Business'})),
    'EDM': ('Engagement & Digital Marketing (Sales)', frozenset({'Business Development', 'Commercial', 'Enterprise', 'Inbound - SDR', 'MM Outbound SDRs', 'Marketing Operations', 'Mid Market', 'Outbound - BDR', 'SDR', 'Sales Development', 'Sales Leadership'})),
    'PM': ('Performance Management', frozenset({'CEO Office', 'Commercial', 'Enterprise', 'Inbound - SDR', 'Leadership', 'Mid Market', 'Outbound - BDR', 'Revenue Operations', 'Sales Leadership', 'Sales Operations'})),
    'CM1': ('Compensation Management', frozenset({'CEO Office', 'Commercial', 'Enterprise', 'Finance', 'Leadership', 'MM', 'Revenue Operations', 'Sales Leadership', 'Sales Operations', 'Small Medium Business'})),
    'CM2': ('Contact Management', frozenset({'Business Development', 'Commercial', 'Enterprise', 'Inbound - SDR', 'MM', 'Outbound - BDR', 'Revenue Operations', 'SDR', 'Sales Leadership', 'Sales Operations'})),
    'CPQ': ('Configure, Price, Quote', frozenset({'CEO Office', 'Commercial', 'Enterprise', 'Finance', 'Leadership', 'Marketing Operations', 'Revenue Operations', 'Sales Leadership', 'Sales Operations', 'Small Medium Business'})),
    'UBM': ('Usage-Based Monetization', frozenset({'Billing & Usage', 'CEO Office', 'Data Analysis', 'Finance', 'Identity, Billing & Usage', 'Leadership', 'Pricing & Monetization Strategy', 'Revenue Operations', 'Sales Leadership', 'Sales Operations'})),
    'PMA': ('Pricing Management', frozenset({'CEO Office', 'Commercial', 'Enterprise', 'Finance', 'Leadership', 'Pricing & Monetization Strategy', 'Revenue Operations', 'Sales Leadership', 'Sales Operations', 'Small Medium Business'})),
    'PCM': ('Product Catalog Management', frozenset({'CEO Office', 'Commercial', 'Enterprise', 'Finance', 'Leadership', 'Marketing Operations', 'Pricing & Monetization Strategy', 'Product Management', 'Product Marketing', 'Revenue Operations', 'Sales Operations'})),
    'QMA': ('Quote Management', frozenset({'CEO Office', 'Commercial', 'Enterprise', 'Finance', 'Leadership', 'Marketing Operations', 'Mid Market', 'Revenue Operations', 'Sales Leadership', 'Sales Operations', 'Small Medium Business'})),
    'GSC': ('Guided Solution Configuration', frozenset({'CEO Office', 'Commercial', 'Enterprise', 'Leadership', 'Pricing & Monetization Strategy', 'Revenue Operations', 'Sales Enablement', 'Sales Leadership', 'Sales Operations', 'Small Medium Business'})),
    'CLM': ('Contract Lifecycle Management', frozenset({'CEO Office', 'Commercial', 'Enterprise', 'Finance', 'Leadership', 'Legal', 'Revenue Operations', 'Sales Leadership', 'Sales Operations', 'Small Medium Business'})),
    'QAP': ('Quote Approval Process', frozenset({'CEO Office', 'Commercial', 'Enterprise', 'Finance', 'Leadership', 'Legal', 'Pricing & Monetization Strategy', 'Revenue Operations', 'Sales Leadership', 'Sales Operations'})),
    'PRC': ('Pricing & Discounting', frozenset({'CEO Office', 'Commercial', 'Enterprise', 'Finance', 'Leadership', 'Pricing & Monetization Strategy', 'Revenue Operations', 'Sales Leadership', 'Sales Operations', 'Small Medium Business'})),
    'CSC': ('Customer Success', frozenset({'CEO Office', 'Customer Care', 'Customer Education', 'Customer Solutions', 'Customer Success', 'Customer Support', 'Leadership', 'Marketing Operations', 'Professional Services', 'Revenue Operations', 'Solutions Architects'})),
    'PSR': ('Proactive Service', frozenset({'CEO Office', 'Customer Care', 'Customer Solutions', 'Customer Success', 'Customer Support', 'Data Analysis', 'Leadership', 'Professional Services', 'Revenue Operations', 'Solutions Architects'})),
    'CSO': ('Customer Solutions & Implementation', frozenset({'Customer Solutions', 'Customer Success', 'Customer Support', 'Expert Services', 'Partners & Alliances', 'Professional Services', 'Solutions Architects', 'Solutions Engineers', 'Solutions Team', 'XS/PS'})),
    'CM3': ('Case Management', frozenset({'Customer Care', 'Customer Solutions', 'Customer Success', 'Customer Support', 'Leadership', 'Marketing Operations', 'Professional Services', 'Solutions Architects', 'Support'})),
    'KBM': ('Knowledge Management', frozenset({'Community', 'Customer Care', 'Customer Education', 'Customer Solutions', 'Customer Support', 'DevRel', 'Developer Relations', 'Product Marketing', 'Support', 'Technical Programs'})),
    'SSE': ('Self Service', frozenset({'Community', 'Customer Care', 'Customer Education', 'Customer Support', 'DevRel', 'Developer Relations', 'Education', 'Leadership', 'Support'})),
    'SAI': ('Service AI & Automation', frozenset({'AI', 'Customer Care', 'Customer Success', 'Customer Support', 'Data Analysis', 'Leadership', 'Solutions Architects', 'Support'})),
    'OCS': ('Omni-channel Service', frozenset({'CEO Office', 'Customer Care', 'Customer Solutions', 'Customer Success', 'Customer Support', 'Leadership', 'Solutions Architects', 'Support'})),
    'SAN': ('Service Analytics', frozenset({'CEO Office', 'Customer Care', 'Customer Success', 'Customer Support', 'Data Analysis', 'Leadership', 'Product Management', 'R&D Leadership', 'Support'})),
    'UAE': ('Unified Agent Experience', frozenset({'Customer Care', 'Customer Success', 'Customer Support', 'Leadership', 'Solutions Architects', 'Support'})),
    'WFM': ('Workforce Management', frozenset({'CEO Office', 'Customer Care', 'Customer Support', 'Data Analysis', 'Leadership', 'Revenue Operations', 'Sales Operations', 'Solutions Architects', 'Support'})),
    'RPL': ('Resource Planning', frozenset({'CEO Office', 'Customer Solutions', 'Expert Services', 'Leadership', 'Professional Services', 'Sales Operations', 'Solutions Architects', 'Solutions Engineers', 'Solutions Team', 'XS/PS'})),
    'TON': ('Training & Onboarding', frozenset({'CEO Office', 'Commercial', 'Customer Care', 'Customer Solutions', 'Customer Support', 'Enterprise', 'Inbound - SDR', 'Leadership', 'Learning & Development', 'Marketing Operations', 'Outbound - BDR'})),
    'PRM': ('Partner Relationship Management', frozenset({'CEO Office', 'Leadership', 'Partner', 'Partners & Alliances', 'Revenue Operations', 'Sales Leadership', 'Sales Operations', 'XS/PS'})),
    'PRO': ('Partner Recruitment & Onboarding', frozenset({'CEO Office', 'Leadership', 'Partner', 'Partners & Alliances', 'Revenue Operations', 'Sales Leadership', 'Sales Operations', 'XS/PS'})),
    'PDR': ('Deal Registration & Co-selling', frozenset({'CEO Office', 'Commercial', 'Enterprise', 'Leadership', 'Partner', 'Partners & Alliances', 'Revenue Operations', 'Sales Leadership', 'Sales Operations', 'XS/PS'})),
    'PEN': ('Partner Enablement', frozenset({'Customer Education', 'Leadership', 'Partner', 'Partners & Alliances', 'Sales Enablement', 'Solutions Architects', 'Solutions Engineers', 'XS/PS'})),
    'MDF': ('MDF & Incentives', frozenset({'CEO Office', 'Data Analysis', 'Finance', 'Leadership', 'Partner', 'Partners & Alliances', 'Revenue Operations', 'Sales Leadership', 'XS/PS'})),
    'PMK': ('Partner Marketing', frozenset({'CEO Office', 'Field Marketing', 'Leadership', 'Marketing Leadership', 'Marketing Operations', 'Partner', 'Partners & Alliances', 'Product Marketing', 'Revenue Operations', 'XS/PS'})),
    'PAN': ('Partner Analytics', frozenset({'CEO Office', 'Data Analysis', 'Leadership', 'Partner', 'Partners & Alliances', 'Revenue Operations', 'Sales Leadership', 'Sales Operations', 'XS/PS'})),
    'PEC': ('Partner Ecosystem', frozenset({'Business Development', 'CEO Office', 'Integrations', 'Leadership', 'Partner', 'Partners & Alliances', 'Revenue Operations', 'XS/PS'})),
    'PSM': ('Partner Sales Management', frozenset({'CEO Office', 'Commercial', 'Enterprise', 'Leadership', 'Partner', 'Partners & Alliances', 'Revenue Operations', 'Sales Leadership', 'Sales Operations', 'XS/PS'})),
    'JPM': ('Joint Partner Marketing', frozenset({'CEO Office', 'Field Marketing', 'Leadership', 'Marketing Leadership', 'Marketing Operations', 'Partner', 'Partners & Alliances', 'Product Marketing', 'Revenue Operations', 'XS/PS'})),
    'PS1': ('Partner Service Management', frozenset({'CEO Office', 'Customer Care', 'Data Analysis', 'Leadership', 'Partner', 'Partners & Alliances', 'Revenue Operations', 'Sales Leadership', 'Sales Operations', 'XS/PS'})),
    'SUB': ('Subscription & Usage Billing', frozenset({'Accounting', 'CEO Office', 'Control', 'Control | Merchandising', 'Customer Solutions', 'Data Analysis', 'Finance', 'Leadership', 'Marketing Operations', 'Pricing & Monetization Strategy', 'Revenue Operations', 'Sales Leadership'})),
    'INV': ('Invoice Management', frozenset({'Accounting', 'CEO Office', 'Data Analysis', 'Finance', 'Leadership', 'Marketing Operations', 'Revenue Operations'})),
    'PMT': ('Payment Processing', frozenset({'Accounting', 'CEO Office', 'Compliance Security', 'Data Analysis', 'Finance', 'Identity, Billing & Usage', 'Leadership', 'Revenue Operations'})),
    'RCG': ('Revenue Recognition', frozenset({'Accounting', 'CEO Office', 'Data Analysis', 'FP&A', 'Finance', 'Leadership', 'Pricing & Monetization Strategy', 'Revenue Operations'})),
    'ARM': ('Accounts Receivable Management', frozenset({'Accounting', 'CEO Office', 'Data Analysis', 'Finance', 'Leadership', 'Revenue Operations', 'Sales Operations'})),
    'TXM': ('Tax Management', frozenset({'Accounting', 'CEO Office', 'Compliance Security', 'Data Analysis', 'Finance', 'Leadership', 'Legal', 'Revenue Operations'})),
    'BAN': ('Billing Analytics', frozenset({'Accounting', 'Billing & Usage', 'CEO Office', 'Data Analysis', 'FP&A', 'Finance', 'Leadership', 'Pricing & Monetization Strategy', 'Revenue Operations', 'Sales Leadership'})),
    'MBL': ('Multi-entity Billing', frozenset({'Accounting', 'CEO Office', 'Data Analysis', 'FP&A', 'Finance', 'Leadership', 'Legal', 'Revenue Operations'})),
    'CSP': ('Customer Self-service Portal', frozenset({'Billing & Usage', 'CEO Office', 'Customer Support', 'Dashboard', 'Data Analysis', 'Finance', 'Identity, Billing & Usage', 'Leadership', 'Pricing & Monetization Strategy', 'Revenue Operations'})),
    'ANL': ('Analytics & Reporting', frozenset({'Business Operations', 'CEO Office', 'Data Analysis', 'FP&A', 'Finance', 'Leadership', 'Revenue Operations', 'Sales Leadership', 'Sales Operations'})),
    'CDM': ('Customer Data Management', frozenset({'Business Operations', 'CEO Office', 'Data Analysis', 'Data Governance', 'Data Platform', 'Leadership', 'Marketing Operations', 'Revenue Operations', 'Sales Operations'})),
    'ORC': ('Order Capture', frozenset({'Accounting', 'Billing & Usage', 'CEO Office', 'Commercial', 'Enterprise', 'Finance', 'Leadership', 'Revenue Operations', 'Sales Operations', 'Small Medium Business'})),
    'SPM': ('Sales Performance Management', frozenset({'CEO Office', 'Commercial', 'Enterprise', 'Finance', 'Leadership', 'MM', 'Marketing Operations', 'Revenue Operations', 'Sales Leadership', 'Sales Operations', 'Small Medium Business'})),
    'PEM': ('Platform Engineering & Site Reliability', frozenset({'Fleet', 'IaaS', 'Infra - Edge', 'Infra - Ops', 'Infrastructure', 'PaaS', 'R&D Leadership', 'Scale', 'Search Infra', 'Technical Programs'})),
    'ACP': ('Automated Code Promotion (CI/CD)', frozenset({'Developer Experience', 'Engineering', 'Engineering Security', 'Foundation', 'Infra - Dev Tooling', 'Product Operations & Development', 'R&D Leadership', 'Technical Programs'})),
    'MBD': ('Microservices Build & Deployment', frozenset({'Engineering', 'Infra - Dev Tooling', 'Infra - General', 'Infra - Ops', 'Infrastructure', 'PaaS', 'Product Operations & Development', 'R&D Leadership', 'Technical Programs'})),
    'VCS': ('Version Control System', frozenset({'Developer Experience', 'Engineering', 'Engineering Security', 'Infra - Dev Tooling', 'Infrastructure', 'R&D Leadership', 'Search Core', 'Search Infra', 'Technical Programs'})),
    'IDM': ('Infrastructure Deployment & Management', frozenset({'CEO Office', 'Engineering Security', 'Fleet', 'IaaS', 'Infra - Dev Tooling', 'Infra - Ops', 'Infrastructure', 'Leadership', 'PaaS', 'R&D Leadership'})),
    'PLS': ('Polyglot Language Support', frozenset({'DX', 'Developer Experience', 'Engineering', 'Foundation', 'IaaS', 'Infra - Dev Tooling', 'Infrastructure', 'PaaS', 'R&D Leadership', 'Technical Programs'})),
    'LCA': ('Low Code / Accelerated Development', frozenset({'Business Operations', 'CEO Office', 'IT', 'Internal Tools', 'Leadership', 'Marketing Operations', 'Revenue Operations', 'Sales Operations', 'Salesforce'})),
    'AUM': ('Automated Upgrade Management', frozenset({'CEO Office', 'Engineering', 'IT', 'Infra - Dev Tooling', 'Infra - Ops', 'Infrastructure', 'Leadership', 'R&D Leadership', 'Security', 'Technical Programs'})),
    'PCO': ('Platform Compliance & Certifications', frozenset({'CEO Office', 'Compliance Security', 'Engineering Security', 'Leadership', 'Legal', 'Security', 'Security Tools'})),
    'IDE': ('Identification (Risk)', frozenset({'CEO Office', 'Compliance Security', 'Engineering Security', 'IT', 'Leadership', 'Legal', 'Security', 'Security Tools'})),
    'SPT': ('Protection', frozenset({'CEO Office', 'Compliance Security', 'Engineering Security', 'IT', 'Infra - Dev Tooling', 'Leadership', 'Security', 'Security Tools'})),
    'DET': ('Detection', frozenset({'CEO Office', 'Compliance Security', 'Engineering Security', 'IT', 'Infra - Ops', 'Infrastructure', 'Leadership', 'Security', 'Security Tools'})),
    'RMA': ('Response Management', frozenset({'CEO Office', 'Compliance Security', 'Corporate Communications', 'Customer Support', 'Engineering Security', 'Infra - Ops', 'Leadership', 'Legal', 'Security', 'Security Tools'})),
    'DSE': ('Data Security', frozenset({'CEO Office', 'Compliance Security', 'Core Api', 'Dashboard', 'Engineering Security', 'Infrastructure', 'Leadership', 'Legal', 'Security', 'Security Tools'})),
    'DWB': ('Data Warehousing, BI & Analytics', frozenset({'Business Operations', 'CEO Office', 'Data', 'Data Analysis', 'Data Analytics', 'Data Governance', 'Data Platform', 'IT', 'Leadership'})),
    'DGO': ('Data Governance', frozenset({'Business Operations', 'CEO Office', 'Compliance Security', 'Data', 'Data Analysis', 'Data Analytics', 'Data Governance', 'Data Platform', 'IT', 'Leadership'})),
    'MDM': ('Master Data Management', frozenset({'Business Operations', 'CEO Office', 'Data Analysis', 'Data Governance', 'Data Platform', 'Finance', 'Leadership', 'Revenue Operations'})),
    'DQU': ('Data Quality', frozenset({'Business Operations', 'CEO Office', 'Data Analysis', 'Data Analytics', 'Data Governance', 'Data Platform', 'Leadership'})),
    'DRH': ('Data Resiliency (HA/DR)', frozenset({'CEO Office', 'Compliance Security', 'Data Platform', 'Fleet', 'IaaS', 'Infrastructure', 'Leadership', 'PaaS', 'R&D Leadership', 'Search Infra'})),
    'DPC': ('Data Policies & Compliance', frozenset({'CEO Office', 'Compliance Security', 'Data Governance', 'Data Platform', 'Engineering Security', 'Finance', 'Leadership', 'Legal', 'Security'})),
    'DST': ('Data Strategy', frozenset({'Business Operations', 'CEO Office', 'Compliance Security', 'Data Analysis', 'Data Analytics', 'Data Governance', 'Data Platform', 'Leadership'})),
    'DDC': ('Dashboards Design & Configuration', frozenset({'Business Operations', 'CEO Office', 'Data Analysis', 'Data Analytics', 'Data Governance', 'Data Platform', 'IT', 'Leadership'})),
    'SSA': ('Self Service Analytics', frozenset({'Business Operations', 'CEO Office', 'Data', 'Data Analysis', 'Data Analytics', 'Data Governance', 'Data Platform', 'IT', 'Leadership'})),
    'UAN': ('Usage Analytics', frozenset({'AI Search', 'AI Search | Ranking', 'CEO Office', 'Dashboard', 'Data Analysis', 'Data Analytics', 'Leadership', 'Pricing & Monetization Strategy', 'Product Management', 'Product Marketing'})),
    'IOR': ('Integration Orchestration', frozenset({'Business Operations', 'CEO Office', 'Data Analysis', 'Data Platform', 'Finance', 'IT', 'Leadership', 'Salesforce'})),
    'EVT': ('Event-Driven Messaging', frozenset({'Billing & Usage', 'Data Analytics', 'Data Platform', 'Events & Records Connection', 'Events & Records Pipeline', 'Identity, Billing & Usage', 'Infrastructure', 'R&D Leadership', 'Scale', 'Technical Programs'})),
    'CCS': ('Cloud Connectivity (SaaS / PaaS)', frozenset({'Business Operations', 'Data Governance', 'Data Platform', 'Finance', 'IT', 'Leadership', 'Revenue Operations', 'Sales Operations', 'Salesforce'})),
    'CCO': ('Contextual Collaboration', frozenset({'CEO Office', 'Commercial', 'Customer Success', 'Customer Support', 'Enterprise', 'Leadership', 'Mid Market', 'Revenue Operations', 'Sales Leadership', 'Sales Operations'})),
    'VCO': ('Video Conferencing', frozenset({'CEO Office', 'Employee Experience', 'Human Resources', 'IT', 'Leadership', 'Operations & Workplace'})),
    'TCO': ('Text Conferencing', frozenset({'CEO Office', 'Employee Experience', 'Human Resources', 'IT', 'Leadership', 'Operations & Workplace'})),
}


def divisions_for(capability_ids: list[str]) -> list[str]:
    """Union of allowed divisions across the given capability IDs (sorted)."""
    out: set[str] = set()
    for cid in capability_ids:
        entry = CAPABILITY_DIVISIONS.get(cid)
        if entry:
            out |= entry[1]
    return sorted(out)


def valid_capability_ids() -> set[str]:
    """All capability IDs that have a division mapping."""
    return set(CAPABILITY_DIVISIONS)
