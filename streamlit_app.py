"""
DC Traffic Safety Survey Dashboard
===================================
A Streamlit dashboard for exploring traffic safety survey data from the DC metropolitan area.

To run:
1. Install requirements: pip install streamlit pandas plotly openpyxl
2. Place your data file in the same directory
3. Run: streamlit run dc_survey_dashboard_v4.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="DC Traffic Safety Survey",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CLEAN LIGHT THEME
# =============================================================================
st.markdown("""
<style>
    /* Force light theme */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Sidebar - green */
    [data-testid="stSidebar"] {
        background-color: #26686d;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stSidebar"] label {
        color: #ffffff !important;
    }
    
    /* Sidebar metrics - white text */
    [data-testid="stSidebar"] [data-testid="metric-container"] * {
        color: #ffffff !important;
        background: transparent !important;
    }
    
    /* Sidebar collapse button */
    [data-testid="stSidebar"] button svg,
    [data-testid="collapsedControl"] svg {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    
    /* Override multiselect pill colors - teal background, white text */
    [data-testid="stSidebar"] span[data-baseweb="tag"] {
        background-color: #26686d !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] span[data-baseweb="tag"] span {
        color: #ffffff !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #666666 !important;
    }
    
    /* Tab styling - ensure visibility */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #f1f5f9;
        padding: 4px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #475569 !important;
        border-radius: 6px;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #666666 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Metric styling - base */
    [data-testid="metric-container"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
    }
    
    /* Override Streamlit's metric value color - slate for main area */
    div[data-testid="stMetricValue"] {
        color: #666666 !important;
    }
    
    /* Keep sidebar metrics white */
    [data-testid="stSidebar"] div[data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    
    /* Main area metrics - slate text */
    [data-testid="stMainBlockContainer"] [data-testid="metric-container"] * {
        color: #666666 !important;
        background: transparent !important;
    }
    
    /* Main content area text - slate gray */
    [data-testid="stMainBlockContainer"] p,
    [data-testid="stMainBlockContainer"] span,
    [data-testid="stMainBlockContainer"] em,
    [data-testid="stMainBlockContainer"] strong,
    [data-testid="stMainBlockContainer"] li {
        color: #666666 !important;
    }
    
    /* Selectbox and input labels */
    .stSelectbox label, .stMultiSelect label, .stRadio label {
        color: #666666 !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data
def load_data():
    """Load and prepare the survey data."""
    df = pd.read_csv('Dc_dash.csv')
    return df

# =============================================================================
# VARIABLE MAPPINGS
# =============================================================================

RACE_VARS = {
    'white': 'White',
    'black': 'Black',
    'latine': 'Latine/Hispanic',
    'asian': 'Asian',
    'multi': 'Multiracial',
    'me': 'Middle Eastern',
    'other': 'Other',
    'pna': 'Prefer not to answer'
}

GENDER_VARS = {
    'malebin': 'Male',
    'femalebin': 'Female',
    'naf': 'Non-binary/Other'
}

YEARS_IN_DC_ORDER = [
    'Less than 6 months',
    '6 months to 1 year', 
    '1–3 years',
    '3–5 years',
    'More than 5 years',
    'Prefer not to answer'
]

DRIVING_BEHAVIOR_VARS = {
    'alcohol': 'Driving within 2 hrs of 3+ drinks',
    'cannabis5': 'Driving within 5 hrs of smoking cannabis',
    'cannabis9': 'Driving within 9 hrs of ingesting cannabis',
    'sim': 'Driving within 2 hrs of alcohol + cannabis',
    'rx': 'Driving while feeling effects of Rx/drugs',
    'text': 'Driving while manually using phone',
    'speed': 'Driving 10+ mph over speed limit',
    'drowsy': 'Driving after less than 5 hrs sleep',
    'seatbelt': 'Driving without seatbelt'
}

DANGER_VARS = {
    'dangeralc': 'Driving within 2 hrs of 3+ drinks',
    'dangercann5': 'Driving within 5 hrs of smoking cannabis',
    'dangercann9': 'Driving within 9 hrs of ingesting cannabis',
    'dangersim': 'Driving within 2 hrs of alcohol + cannabis',
    'dangerrx': 'Driving while feeling effects of Rx/drugs',
    'dangertext': 'Driving while manually using phone',
    'dangerspeed': 'Driving 10+ mph over speed limit',
    'dangerdrowsy': 'Driving after less than 5 hrs sleep',
    'dangerseatbelt': 'Driving without seatbelt'
}

LEGAL_VARS = {
    'legalalc': 'Driving within 2 hrs of 3+ drinks',
    'legalcann5': 'Driving within 5 hrs of smoking cannabis',
    'legalcann9': 'Driving within 9 hrs of ingesting cannabis',
    'legalsim': 'Driving within 2 hrs of alcohol + cannabis',
    'legalrx': 'Driving while feeling effects of Rx/drugs',
    'legaltext': 'Driving while manually using phone',
    'legalspeed': 'Driving 10+ mph over speed limit',
    'legaldrowsy': 'Driving after less than 5 hrs sleep',
    'legalseatbelt': 'Driving without seatbelt'
}

NORMS_VARS = {
    'normsalc': 'Driving within 2 hrs of 3+ drinks',
    'normscann5': 'Driving within 5 hrs of smoking cannabis',
    'normscann9': 'Driving within 9 hrs of ingesting cannabis',
    'normssim': 'Driving within 2 hrs of alcohol + cannabis',
    'normsrx': 'Driving while feeling effects of Rx/drugs',
    'normstext': 'Driving while manually using phone',
    'normsspeed': 'Driving 10+ mph over speed limit',
    'normsdrowsy': 'Driving after less than 5 hrs sleep',
    'normsbelt': 'Driving without seatbelt'
}

AVOIDANCE_VARS = {
    'DD': 'Designated driver',
    'rideshare': 'Rideshare (Uber/Lyft)',
    'walk': 'Walking',
    'stayover': 'Staying overnight',
    'publictransit': 'Public transit',
    'bikes': 'Biking/scootering'
}

# Behavior Comparison tab variable mappings (using same vars as rest of dashboard)
BEHAVIOR_COMPARISON_VARS = {
    'alcohol': {
        'label': 'Driving within 2 hrs of 3+ drinks',
        'prevalence': 'alcohol',
        'danger': 'dangeralc',
        'enforcement': 'legalalc',
        'norms': 'normsalc'
    },
    'text': {
        'label': 'Driving while manually using phone',
        'prevalence': 'text',
        'danger': 'dangertext',
        'enforcement': 'legaltext',
        'norms': 'normstext'
    },
    'drowsy': {
        'label': 'Driving after less than 5 hrs sleep',
        'prevalence': 'drowsy',
        'danger': 'dangerdrowsy',
        'enforcement': 'legaldrowsy',
        'norms': 'normsdrowsy'
    },
    'speed': {
        'label': 'Driving 10+ mph over speed limit',
        'prevalence': 'speed',
        'danger': 'dangerspeed',
        'enforcement': 'legalspeed',
        'norms': 'normsspeed'
    },
    'cannabis9': {
        'label': 'Driving within 9 hrs of ingesting cannabis',
        'prevalence': 'cannabis9',
        'danger': 'dangercann9',
        'enforcement': 'legalcann9',
        'norms': 'normscann9'
    },
    'sim': {
        'label': 'Driving within 2 hrs of alcohol + cannabis',
        'prevalence': 'sim',
        'danger': 'dangersim',
        'enforcement': 'legalsim',
        'norms': 'normssim'
    },
    'rx': {
        'label': 'Driving while feeling effects of Rx/drugs',
        'prevalence': 'rx',
        'danger': 'dangerrx',
        'enforcement': 'legalrx',
        'norms': 'normsrx'
    },
    'seatbelt': {
        'label': 'Driving without seatbelt',
        'prevalence': 'seatbelt',
        'danger': 'dangerseatbelt',
        'enforcement': 'legalseatbelt',
        'norms': 'normsbelt'
    }
}

FREQUENCY_ORDER = ['Never', 'Once', 'Twice', 'More than twice', 'Prefer not to answer']
# Display order for charts (strongest response first)
DANGER_ORDER = ['Very dangerous', 'Somewhat dangerous', 'Unsure', 'Somewhat safe', 'Very safe', 'Prefer not to answer']
LIKELIHOOD_ORDER = ['Very likely', 'Somewhat likely', 'Unsure', 'Somewhat unlikely', 'Very unlikely', 'Prefer not to answer']
HELMET_ORDER = ['None', 'A Few', 'Half', 'All', 'Prefer not to answer']
ELECTRIC_ORDER = ['None', 'A Few', 'Half', 'All', 'Prefer not to answer']

# =============================================================================
# INSTITUTION COLOR PALETTE
# =============================================================================
COLORS = {
    'primary': '#26686d',       # Teal
    'secondary': '#5d1542',     # Burgundy
    'accent': '#dcaa38',        # Gold
    'neutral': '#666666',       # Gray
    'warm': '#5d1542',          # Burgundy (alias)
    'gold': '#dcaa38',          # Gold (alias)
    'text': '#666666',          # Slate text
    'background': '#ffffff',    # White
    'border': '#e2e8f0',        # Light border
    # Categorical palette - using institution colors
    'categorical': ['#26686d', '#5d1542', '#dcaa38', '#666666', '#3d8a8f', '#7d3562'],
    # Sequential for frequency (teal = good/never, burgundy = bad/frequent)
    'frequency': ['#26686d', '#666666', '#dcaa38', '#5d1542', '#e2e8f0'],
    # For danger/likelihood scales (teal=safe, gray=unsure, burgundy=dangerous)
    'diverging': ['#26686d', '#4a9a9f', '#666666', '#dcaa38', '#5d1542', '#e2e8f0'],
}

# Standard chart layout with border
CHART_LAYOUT = {
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'paper_bgcolor': '#ffffff',
    'font': {'color': '#666666', 'size': 13},
    'margin': dict(l=20, r=20, t=50, b=20),
    'shapes': [{
        'type': 'rect',
        'xref': 'paper',
        'yref': 'paper',
        'x0': 0,
        'y0': 0,
        'x1': 1,
        'y1': 1,
        'line': {'color': '#e2e8f0', 'width': 1}
    }]
}

# Plotly template for consistent styling
PLOT_LAYOUT = {
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'font': {'color': '#666666', 'family': 'system-ui, -apple-system, sans-serif', 'size': 13},
    'title': {'font': {'size': 16, 'color': '#666666'}},
    'xaxis': {'tickfont': {'color': '#334155', 'size': 12}, 'title': {'font': {'color': '#334155', 'size': 13}}},
    'yaxis': {'tickfont': {'color': '#334155', 'size': 12}, 'title': {'font': {'color': '#334155', 'size': 13}}},
    'margin': dict(l=20, r=20, t=50, b=20),
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_gender_column(df):
    """Create a single gender column from binary columns."""
    df = df.copy()
    df['gender'] = 'Unknown'
    df.loc[df['malebin'] == 1, 'gender'] = 'Male'
    df.loc[df['femalebin'] == 1, 'gender'] = 'Female'
    df.loc[df['naf'] == 1, 'gender'] = 'Non-binary/Other'
    return df

def filter_dataframe(df, filters):
    """Apply sidebar filters to dataframe."""
    filtered = df.copy()
    
    if filters['age']:
        filtered = filtered[filtered['age'].isin(filters['age'])]
    if filters['gender']:
        filtered = filtered[filtered['gender'].isin(filters['gender'])]
    if filters['city']:
        filtered = filtered[filtered['city'].isin(filters['city'])]
    if filters['enrollment']:
        filtered = filtered[filtered['enroll'].isin(filters['enrollment'])]
    if filters['university']:
        filtered = filtered[filtered['university'].isin(filters['university'])]
    if filters['driver_status'] != 'All':
        if filters['driver_status'] == 'Drivers only':
            filtered = filtered[filtered['drive'] == 'Yes']
        else:
            filtered = filtered[filtered['drive'] == 'No']
    
    return filtered

def calculate_prevalence(df, var, exclude_pna=True):
    """Calculate prevalence (% who did behavior at least once)."""
    if var not in df.columns:
        return None
    
    counts = df[var].value_counts()
    total = counts.sum()
    
    if exclude_pna and 'Prefer not to answer' in counts.index:
        total = total - counts.get('Prefer not to answer', 0)
        counts = counts.drop('Prefer not to answer', errors='ignore')
    
    if total == 0:
        return 0
    
    ever = total - counts.get('Never', 0)
    return (ever / total) * 100

def calculate_prevalence_binary(df, var):
    """Calculate prevalence for binary (0/1) variables."""
    if var not in df.columns:
        return None
    valid = df[var].dropna()
    if len(valid) == 0:
        return None
    return (valid.sum() / len(valid)) * 100

def calculate_danger_pct(df, var):
    """Calculate % rating as somewhat or very dangerous (scores 4-5 on 1-5 scale)."""
    if var not in df.columns:
        return None
    valid = df[var].dropna()
    if len(valid) == 0:
        return None
    # 4 = Somewhat dangerous, 5 = Very dangerous
    dangerous = ((valid >= 4).sum() / len(valid)) * 100
    return dangerous

def calculate_enforcement_pct(df, var):
    """Calculate % rating as somewhat or very likely (scores 4-5 on 1-5 scale)."""
    if var not in df.columns:
        return None
    valid = df[var].dropna()
    if len(valid) == 0:
        return None
    # 4 = Somewhat likely, 5 = Very likely
    likely = ((valid >= 4).sum() / len(valid)) * 100
    return likely

def calculate_distribution(df, var, order=None):
    """Calculate frequency distribution for a variable."""
    if var not in df.columns:
        return pd.DataFrame()
    
    counts = df[var].value_counts()
    pcts = (counts / counts.sum() * 100).round(1)
    
    result = pd.DataFrame({
        'Category': pcts.index,
        'Percentage': pcts.values,
        'Count': counts.values
    })
    
    if order:
        result['Category'] = pd.Categorical(result['Category'], categories=order, ordered=True)
        result = result.sort_values('Category').dropna(subset=['Category'])
    
    return result

def create_prevalence_chart(df, var_dict, title, color=None):
    """Create horizontal bar chart showing prevalence rates."""
    if color is None:
        color = COLORS['primary']
    
    data = []
    for var, label in var_dict.items():
        prev = calculate_prevalence(df, var)
        if prev is not None:
            data.append({'Behavior': label, 'Prevalence': prev})
    
    if not data:
        return None
    
    chart_df = pd.DataFrame(data).sort_values('Prevalence', ascending=True)
    n = len(df)
    
    fig = px.bar(
        chart_df,
        x='Prevalence',
        y='Behavior',
        orientation='h',
        text=chart_df['Prevalence'].apply(lambda x: f'{x:.1f}%'),
        color_discrete_sequence=[color]
    )
    
    fig.update_traces(
        textposition='outside', 
        textfont=dict(color='#000000', size=12)
    )
    fig.update_layout(
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font={'color': '#666666', 'size': 13},
        title={'text': f'{title} (N={n})', 'font': {'size': 16, 'color': '#666666'}},
        xaxis_title='% who engaged at least once (past 30 days)',
        yaxis_title='',
        xaxis=dict(range=[0, 100], tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, showgrid=True, gridcolor='#e2e8f0', gridwidth=1),
        yaxis=dict(tickfont={'color': '#666666', 'size': 12}, showgrid=True, gridcolor='#e2e8f0', gridwidth=1),
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        shapes=[{
            'type': 'rect', 'xref': 'paper', 'yref': 'paper',
            'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1,
            'line': {'color': '#e2e8f0', 'width': 1}
        }]
    )
    
    return fig

def create_distribution_chart(df, var, title, order=None, colors=None):
    """Create bar chart showing response distribution."""
    dist = calculate_distribution(df, var, order)
    
    if dist.empty:
        return None
    
    if colors is None:
        colors = COLORS['categorical']
    
    n = dist['Count'].sum()
    
    fig = px.bar(
        dist,
        x='Category',
        y='Percentage',
        text=dist['Percentage'].apply(lambda x: f'{x:.1f}%'),
        color='Category',
        color_discrete_sequence=colors
    )
    
    fig.update_traces(textposition='outside', showlegend=False, textfont=dict(color='#000000', size=12))
    title_text = f'{title} (N={n})' if title else f'(N={n})'
    fig.update_layout(
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font={'color': '#666666', 'size': 13},
        title={'text': title_text, 'font': {'size': 16, 'color': '#666666'}},
        xaxis_title='',
        yaxis_title='Percentage',
        xaxis=dict(tickfont={'color': '#666666', 'size': 12}, showgrid=True, gridcolor='#e2e8f0'),
        yaxis=dict(range=[0, max(dist['Percentage']) * 1.25], tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, showgrid=True, gridcolor='#e2e8f0'),
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        shapes=[{
            'type': 'rect', 'xref': 'paper', 'yref': 'paper',
            'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1,
            'line': {'color': '#e2e8f0', 'width': 1}
        }]
    )
    
    return fig

def create_crosstab_chart(df, behavior_var, demo_var, demo_label, behavior_label):
    """Create bar chart for cross-tabulation."""
    if behavior_var not in df.columns or demo_var not in df.columns:
        return None
    
    # Exclude "Prefer not to answer" and "Non-binary/Other" from cross-tabs
    exclude_values = ['Prefer not to answer']
    if demo_var == 'gender':
        exclude_values.append('Non-binary/Other')
    
    data = []
    for group in df[demo_var].dropna().unique():
        if group in exclude_values:
            continue
        subset = df[df[demo_var] == group]
        prev = calculate_prevalence(subset, behavior_var)
        if prev is not None:
            data.append({'Group': str(group), 'Prevalence': prev, 'N': len(subset)})
    
    if not data:
        return None
    
    chart_df = pd.DataFrame(data)
    total_n = chart_df['N'].sum()
    
    fig = px.bar(
        chart_df,
        x='Group',
        y='Prevalence',
        text=chart_df['Prevalence'].apply(lambda x: f'{x:.1f}%'),
        color_discrete_sequence=[COLORS['primary']]
    )
    
    fig.update_traces(textposition='outside', textfont=dict(color='#000000', size=12))
    fig.update_layout(
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font={'color': '#666666', 'size': 13},
        title={'text': f'{behavior_label} by {demo_label} (N={total_n})', 'font': {'size': 16, 'color': '#666666'}},
        xaxis_title=demo_label,
        yaxis_title='% engaged at least once',
        xaxis=dict(tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, showgrid=True, gridcolor='#e2e8f0'),
        yaxis=dict(range=[0, 100], tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, showgrid=True, gridcolor='#e2e8f0'),
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        shapes=[{
            'type': 'rect', 'xref': 'paper', 'yref': 'paper',
            'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1,
            'line': {'color': '#e2e8f0', 'width': 1}
        }]
    )
    
    return fig

def create_stacked_perception_chart(df, var_dict, title, order, colors):
    """Create stacked horizontal bar chart for perception items - COLLAPSED into binary categories."""
    data = []
    
    # Determine if this is danger or likelihood based on order
    is_danger = 'Very dangerous' in order
    
    for var, label in var_dict.items():
        if var not in df.columns:
            continue
        
        counts = df[var].value_counts()
        total = counts.sum()
        
        # Exclude PNA from calculation
        if 'Prefer not to answer' in counts.index:
            total = total - counts.get('Prefer not to answer', 0)
        
        if total == 0:
            continue
        
        if is_danger:
            # Collapse to Dangerous vs Safe vs Unsure
            dangerous = counts.get('Very dangerous', 0) + counts.get('Somewhat dangerous', 0)
            safe = counts.get('Very safe', 0) + counts.get('Somewhat safe', 0)
            unsure = counts.get('Unsure', 0)
            
            data.append({'Behavior': label, 'Response': 'Dangerous', 'Percentage': (dangerous / total) * 100})
            data.append({'Behavior': label, 'Response': 'Unsure', 'Percentage': (unsure / total) * 100})
            data.append({'Behavior': label, 'Response': 'Safe', 'Percentage': (safe / total) * 100})
        else:
            # Collapse to Likely vs Unlikely vs Unsure
            likely = counts.get('Very likely', 0) + counts.get('Somewhat likely', 0)
            unlikely = counts.get('Very unlikely', 0) + counts.get('Somewhat unlikely', 0)
            unsure = counts.get('Unsure', 0)
            
            data.append({'Behavior': label, 'Response': 'Likely', 'Percentage': (likely / total) * 100})
            data.append({'Behavior': label, 'Response': 'Unsure', 'Percentage': (unsure / total) * 100})
            data.append({'Behavior': label, 'Response': 'Unlikely', 'Percentage': (unlikely / total) * 100})
    
    if not data:
        return None
    
    chart_df = pd.DataFrame(data)
    n = len(df)
    
    # Set category order - display order matches new DANGER_ORDER/LIKELIHOOD_ORDER (strongest first)
    if is_danger:
        cat_order = ['Dangerous', 'Unsure', 'Safe']
        color_map = {'Dangerous': '#5d1542', 'Unsure': '#666666', 'Safe': '#26686d'}
    else:
        cat_order = ['Likely', 'Unsure', 'Unlikely']
        color_map = {'Likely': '#5d1542', 'Unsure': '#666666', 'Unlikely': '#26686d'}
    
    chart_df['Response'] = pd.Categorical(chart_df['Response'], categories=cat_order, ordered=True)
    
    fig = px.bar(
        chart_df,
        x='Percentage',
        y='Behavior',
        color='Response',
        orientation='h',
        color_discrete_map=color_map,
        category_orders={'Response': cat_order}
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='#ffffff',
        font={'color': '#666666', 'size': 13},
        title={'text': f'{title} (N={n})', 'font': {'size': 16, 'color': '#666666'}},
        xaxis_title='Percentage',
        yaxis_title='',
        xaxis=dict(tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}),
        yaxis=dict(tickfont={'color': '#666666', 'size': 12}),
        barmode='stack',
        height=500,
        legend=dict(
             orientation='h', 
             yanchor='bottom', 
              y=-0.25, 
             xanchor='center', 
             x=0.5,
             font={'color': '#666666', 'size': 12},
             title_text=''
        ),
        margin=dict(l=20, r=20, t=50, b=80),
        shapes=[{
            'type': 'rect', 'xref': 'paper', 'yref': 'paper',
            'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1,
            'line': {'color': '#e2e8f0', 'width': 1}
        }]
    )
    
    return fig

def create_behavior_comparison_chart(df, behavior_key, behavior_vars):
    """Create a bar chart showing 4 aggregate measures for a single behavior."""
    vars_info = behavior_vars[behavior_key]
    label = vars_info['label']
    
    data = []
    
    # 1. Prevalence (% who did behavior at least once) - using existing calculate_prevalence
    prev = calculate_prevalence(df, vars_info['prevalence'])
    if prev is not None:
        data.append({'Measure': 'Prevalence', 'Percentage': prev})
    
    # 2. Danger (% rating as somewhat or very dangerous) - categorical text
    danger_var = vars_info['danger']
    if danger_var in df.columns:
        danger_counts = df[danger_var].value_counts()
        total = danger_counts.sum()
        if 'Prefer not to answer' in danger_counts.index:
            total = total - danger_counts.get('Prefer not to answer', 0)
        if total > 0:
            dangerous = danger_counts.get('Very dangerous', 0) + danger_counts.get('Somewhat dangerous', 0)
            data.append({'Measure': 'Rated Dangerous', 'Percentage': (dangerous / total) * 100})
    
    # 3. Enforcement (% rating as somewhat or very likely) - categorical text
    enforce_var = vars_info['enforcement']
    if enforce_var in df.columns:
        enforce_counts = df[enforce_var].value_counts()
        total = enforce_counts.sum()
        if 'Prefer not to answer' in enforce_counts.index:
            total = total - enforce_counts.get('Prefer not to answer', 0)
        if total > 0:
            likely = enforce_counts.get('Very likely', 0) + enforce_counts.get('Somewhat likely', 0)
            data.append({'Measure': 'Rated Likely (Enforcement)', 'Percentage': (likely / total) * 100})
    
    # 4. Norms (% believing peers engage at least once) - using existing calculate_prevalence
    norms = calculate_prevalence(df, vars_info['norms'])
    if norms is not None:
        data.append({'Measure': 'Peers Engage', 'Percentage': norms})
    
    if not data:
        return None
    
    chart_df = pd.DataFrame(data)
    n = len(df)
    
    # Assign colors to each measure
    color_map = {
        'Prevalence': COLORS['primary'],
        'Rated Dangerous': COLORS['secondary'],
        'Rated Likely (Enforcement)': COLORS['accent'],
        'Peers Engage': COLORS['neutral']
    }
    
    fig = px.bar(
        chart_df,
        x='Measure',
        y='Percentage',
        text=chart_df['Percentage'].apply(lambda x: f'{x:.1f}%'),
        color='Measure',
        color_discrete_map=color_map
    )
    
    fig.update_traces(textposition='outside', showlegend=False, textfont=dict(color='#000000', size=12))
    fig.update_layout(
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font={'color': '#666666', 'size': 13},
        title={'text': f'{label} (N={n})', 'font': {'size': 16, 'color': '#666666'}},
        xaxis_title='',
        yaxis_title='Percentage',
        xaxis=dict(tickfont={'color': '#666666', 'size': 12}, showgrid=True, gridcolor='#e2e8f0'),
        yaxis=dict(range=[0, 100], tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, showgrid=True, gridcolor='#e2e8f0'),
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        shapes=[{
            'type': 'rect', 'xref': 'paper', 'yref': 'paper',
            'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1,
            'line': {'color': '#e2e8f0', 'width': 1}
        }]
    )
    
    return fig

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    df = load_data()
    df = create_gender_column(df)
    
    # =========================================================================
    # SIDEBAR
    # =========================================================================
    with st.sidebar:
        st.title("DC Traffic Safety")
        st.markdown("---")
        st.subheader("Filters")
        
        age_options = sorted([int(x) for x in df['age'].dropna().unique()])
        selected_ages = st.multiselect("Age", options=age_options, default=age_options)
        
        gender_options = ['Male', 'Female', 'Non-binary/Other']
        selected_genders = st.multiselect("Gender", options=gender_options, default=gender_options)
        
        city_options = df['city'].dropna().unique().tolist()
        selected_cities = st.multiselect("Location", options=city_options, default=city_options)
        
        enroll_options = df['enroll'].dropna().unique().tolist()
        selected_enrollment = st.multiselect("College Enrollment", options=enroll_options, default=enroll_options)
        
        # University filter
        if 'university' in df.columns:
            university_options = df['university'].dropna().unique().tolist()
            selected_universities = st.multiselect("University", options=university_options, default=university_options)
        else:
            selected_universities = []
        
        driver_status = st.radio("Driver Status", options=['All', 'Drivers only', 'Non-drivers only'], index=0)
        
        st.markdown("---")
        
        filters = {
            'age': selected_ages,
            'gender': selected_genders,
            'city': selected_cities,
            'enrollment': selected_enrollment,
            'university': selected_universities,
            'driver_status': driver_status
        }
        filtered_df = filter_dataframe(df, filters)
        
        st.metric("Filtered Sample", len(filtered_df))
        st.metric("Total Sample", len(df))
    
    # =========================================================================
    # MAIN CONTENT
    # =========================================================================
    st.title("DC Metropolitan Area Traffic Safety Survey")
    st.markdown("Survey results on driving behaviors, perceptions, and safety practices among young adults (18-24) in the DC area.")
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Demographics",
        "Risky Driving",
        "Danger Perceptions",
        "Enforcement Perceptions",
        "Peer Norms",
        "DUI Avoidance",
        "Micromobility",
        "Behavior Comparison"
    ])
    
    # =========================================================================
    # TAB 1: DEMOGRAPHICS
    # =========================================================================
    with tab1:
        st.header("Demographics Overview")
        st.markdown(f"**{len(filtered_df)} respondents** based on current filters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            age_dist = filtered_df['age'].value_counts().sort_index()
            age_pct = (age_dist / age_dist.sum() * 100)
            fig_age = px.bar(
                x=age_pct.index.astype(int),
                y=age_pct.values,
                labels={'x': 'Age', 'y': 'Percentage'},
                text=age_pct.apply(lambda x: f'{x:.1f}%'),
                color_discrete_sequence=[COLORS['primary']]
            )
            fig_age.update_traces(textposition='outside', textfont=dict(color='#000000', size=12))
            fig_age.update_layout(
                title={'text': f'Age Distribution (N={len(filtered_df)})', 'font': {'size': 16, 'color': '#666666'}},
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font={'color': '#666666', 'size': 13},
                xaxis=dict(tickmode='linear', tick0=18, dtick=1, tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, showgrid=True, gridcolor='#e2e8f0'),
                yaxis=dict(tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, range=[0, max(age_pct.values) * 1.2], showgrid=True, gridcolor='#e2e8f0'),
                height=350,
                margin=dict(l=20, r=20, t=50, b=20),
                shapes=[{'type': 'rect', 'xref': 'paper', 'yref': 'paper', 'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1, 'line': {'color': '#e2e8f0', 'width': 1}}]
            )
            st.plotly_chart(fig_age, use_container_width=True)
            
            # Location
            city_dist = filtered_df['city'].value_counts()
            city_pct = (city_dist / city_dist.sum() * 100)
            fig_city = px.bar(
                x=city_pct.index,
                y=city_pct.values,
                labels={'x': 'Location', 'y': 'Percentage'},
                text=city_pct.apply(lambda x: f'{x:.1f}%'),
                color_discrete_sequence=[COLORS['primary']]
            )
            fig_city.update_traces(textposition='outside', textfont=dict(color='#000000', size=12))
            fig_city.update_layout(
                title={'text': f'Location (N={len(filtered_df)})', 'font': {'size': 16, 'color': '#666666'}},
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font={'color': '#666666', 'size': 13},
                xaxis=dict(tickfont={'color': '#666666', 'size': 12}, title={'text': ''}, showgrid=True, gridcolor='#e2e8f0'),
                yaxis=dict(tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, range=[0, max(city_pct.values) * 1.2], showgrid=True, gridcolor='#e2e8f0'),
                height=350,
                margin=dict(l=20, r=20, t=50, b=20),
                shapes=[{'type': 'rect', 'xref': 'paper', 'yref': 'paper', 'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1, 'line': {'color': '#e2e8f0', 'width': 1}}]
            )
            st.plotly_chart(fig_city, use_container_width=True)
        
        with col2:
            # Gender
            gender_dist = filtered_df['gender'].value_counts()
            gender_pct = (gender_dist / gender_dist.sum() * 100)
            fig_gender = px.bar(
                x=gender_pct.index,
                y=gender_pct.values,
                labels={'x': 'Gender', 'y': 'Percentage'},
                text=gender_pct.apply(lambda x: f'{x:.1f}%'),
                color_discrete_sequence=[COLORS['primary']]
            )
            fig_gender.update_traces(textposition='outside', textfont=dict(color='#000000', size=12))
            fig_gender.update_layout(
                title={'text': f'Gender (N={len(filtered_df)})', 'font': {'size': 16, 'color': '#666666'}},
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font={'color': '#666666', 'size': 13},
                xaxis=dict(tickfont={'color': '#666666', 'size': 12}, title={'text': ''}, showgrid=True, gridcolor='#e2e8f0'),
                yaxis=dict(tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, range=[0, max(gender_pct.values) * 1.2], showgrid=True, gridcolor='#e2e8f0'),
                height=350,
                margin=dict(l=20, r=20, t=50, b=20),
                shapes=[{'type': 'rect', 'xref': 'paper', 'yref': 'paper', 'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1, 'line': {'color': '#e2e8f0', 'width': 1}}]
            )
            st.plotly_chart(fig_gender, use_container_width=True)
            
            # Enrollment
            enroll_dist = filtered_df['enroll'].value_counts()
            enroll_pct = (enroll_dist / enroll_dist.sum() * 100)
            fig_enroll = px.bar(
                x=enroll_pct.index,
                y=enroll_pct.values,
                labels={'x': 'Enrollment', 'y': 'Percentage'},
                text=enroll_pct.apply(lambda x: f'{x:.1f}%'),
                color_discrete_sequence=[COLORS['primary']]
            )
            fig_enroll.update_traces(textposition='outside', textfont=dict(color='#000000', size=12))
            fig_enroll.update_layout(
                title={'text': f'College Enrollment (N={len(filtered_df)})', 'font': {'size': 16, 'color': '#666666'}},
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font={'color': '#666666', 'size': 13},
                xaxis=dict(tickfont={'color': '#666666', 'size': 12}, title={'text': ''}, showgrid=True, gridcolor='#e2e8f0'),
                yaxis=dict(tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, range=[0, max(enroll_pct.values) * 1.2], showgrid=True, gridcolor='#e2e8f0'),
                height=350,
                margin=dict(l=20, r=20, t=50, b=20),
                shapes=[{'type': 'rect', 'xref': 'paper', 'yref': 'paper', 'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1, 'line': {'color': '#e2e8f0', 'width': 1}}]
            )
            st.plotly_chart(fig_enroll, use_container_width=True)
        
        # Race/ethnicity
        st.markdown("*Respondents could select multiple race/ethnicity categories*")
        
        race_data = []
        for var, label in RACE_VARS.items():
            if var in filtered_df.columns:
                pct = (filtered_df[var].sum() / len(filtered_df) * 100)
                race_data.append({'Race/Ethnicity': label, 'Percentage': pct})
        
        race_df = pd.DataFrame(race_data).sort_values('Percentage', ascending=True)
        fig_race = px.bar(
            race_df,
            x='Percentage',
            y='Race/Ethnicity',
            orientation='h',
            text=race_df['Percentage'].apply(lambda x: f'{x:.1f}%'),
            color_discrete_sequence=[COLORS['primary']]
        )
        fig_race.update_traces(textposition='outside', textfont_color='#000000')
        fig_race.update_layout(
            title={'text': f'Race/Ethnicity (N={len(filtered_df)})', 'font': {'size': 16, 'color': '#666666'}},
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font={'color': '#666666', 'size': 13},
            xaxis=dict(range=[0, 60], tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, showgrid=True, gridcolor='#e2e8f0'),
            yaxis=dict(tickfont={'color': '#666666', 'size': 12}, showgrid=True, gridcolor='#e2e8f0'),
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            shapes=[{'type': 'rect', 'xref': 'paper', 'yref': 'paper', 'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1, 'line': {'color': '#e2e8f0', 'width': 1}}]
        )
        st.plotly_chart(fig_race, use_container_width=True)
        
        # Years in DC and Driver status
        col3, col4 = st.columns(2)
        
        with col3:
            years_dist = calculate_distribution(filtered_df, 'yearsindc', YEARS_IN_DC_ORDER)
            if not years_dist.empty:
                fig_years = px.bar(
                    years_dist,
                    x='Category',
                    y='Percentage',
                    text=years_dist['Percentage'].apply(lambda x: f'{x:.1f}%'),
                    color_discrete_sequence=[COLORS['primary']]
                )
                fig_years.update_traces(textposition='outside', textfont_color='#000000')
                fig_years.update_layout(
                    title={'text': f'Years in DC Metro Area (N={len(filtered_df)})', 'font': {'size': 16, 'color': '#666666'}},
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                    font={'color': '#666666', 'size': 13},
                    xaxis=dict(title='', tickfont={'color': '#666666', 'size': 10}, tickangle=-45, showgrid=True, gridcolor='#e2e8f0'),
                    yaxis=dict(tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, range=[0, max(years_dist['Percentage']) * 1.2], showgrid=True, gridcolor='#e2e8f0'),
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=100),
                    shapes=[{'type': 'rect', 'xref': 'paper', 'yref': 'paper', 'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1, 'line': {'color': '#e2e8f0', 'width': 1}}]
                )
                st.plotly_chart(fig_years, use_container_width=True)
        
        with col4:
            drive_dist = filtered_df['drive'].value_counts()
            drive_pct = (drive_dist / drive_dist.sum() * 100)
            fig_drive = px.bar(
                x=drive_pct.index,
                y=drive_pct.values,
                labels={'x': 'Driven', 'y': 'Percentage'},
                text=drive_pct.apply(lambda x: f'{x:.1f}%'),
                color_discrete_sequence=[COLORS['primary']]
            )
            fig_drive.update_traces(textposition='outside', textfont_color='#000000')
            fig_drive.update_layout(
                title={'text': f'Driven in Last 30 Days (N={len(filtered_df)})', 'font': {'size': 16, 'color': '#666666'}},
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font={'color': '#666666', 'size': 13},
                xaxis=dict(tickfont={'color': '#666666', 'size': 12}, title={'text': ''}, showgrid=True, gridcolor='#e2e8f0'),
                yaxis=dict(tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, range=[0, max(drive_pct.values) * 1.2], showgrid=True, gridcolor='#e2e8f0'),
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
                shapes=[{'type': 'rect', 'xref': 'paper', 'yref': 'paper', 'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1, 'line': {'color': '#e2e8f0', 'width': 1}}]
            )
            st.plotly_chart(fig_drive, use_container_width=True)
    
    # =========================================================================
    # TAB 2: RISKY DRIVING
    # =========================================================================
    with tab2:
        st.header("Risky Driving Behavior Prevalence")
        
        drivers_df = filtered_df[filtered_df['drive'] == 'Yes']
        st.markdown(f"**{len(drivers_df)} drivers** based on current filters")
        
        if len(drivers_df) == 0:
            st.warning("No drivers in the current filter selection.")
        else:
            fig_prev = create_prevalence_chart(
                drivers_df,
                DRIVING_BEHAVIOR_VARS,
                'Prevalence of Risky Driving Behaviors (Past 30 Days)',
                color=COLORS['primary']
            )
            if fig_prev:
                st.plotly_chart(fig_prev, use_container_width=True)
            
            st.subheader("Cross-tabulation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                selected_behavior = st.selectbox(
                    "Select behavior:",
                    options=list(DRIVING_BEHAVIOR_VARS.keys()),
                    format_func=lambda x: DRIVING_BEHAVIOR_VARS[x]
                )
            
            with col2:
                demo_options = {'gender': 'Gender', 'city': 'Location', 'enroll': 'Enrollment', 'age': 'Age'}
                selected_demo = st.selectbox(
                    "Cross-tab by:",
                    options=list(demo_options.keys()),
                    format_func=lambda x: demo_options[x]
                )
            
            fig_crosstab = create_crosstab_chart(
                drivers_df, selected_behavior, selected_demo,
                demo_options[selected_demo], DRIVING_BEHAVIOR_VARS[selected_behavior]
            )
            if fig_crosstab:
                st.plotly_chart(fig_crosstab, use_container_width=True)
            
            st.subheader(f"Response Distribution: {DRIVING_BEHAVIOR_VARS[selected_behavior]}")
            fig_dist = create_distribution_chart(
                drivers_df, selected_behavior, '',
                order=FREQUENCY_ORDER, colors=COLORS['frequency']
            )
            if fig_dist:
                st.plotly_chart(fig_dist, use_container_width=True)
    
    # =========================================================================
    # TAB 3: DANGER PERCEPTIONS
    # =========================================================================
    with tab3:
        st.header("Danger Perceptions")
        st.markdown(f"**{len(filtered_df)} respondents** | *How dangerous do you feel it is to...*")
        
        fig_danger = create_stacked_perception_chart(
            filtered_df, DANGER_VARS,
            'Perceived Danger of Risky Driving Behaviors',
            DANGER_ORDER, COLORS['diverging']
        )
        if fig_danger:
            st.plotly_chart(fig_danger, use_container_width=True)
        
        st.subheader("Cross-tabulation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_danger = st.selectbox(
                "Select item:",
                options=list(DANGER_VARS.keys()),
                format_func=lambda x: DANGER_VARS[x],
                key='danger_select'
            )
        
        with col2:
            demo_options = {'gender': 'Gender', 'city': 'Location', 'enroll': 'Enrollment', 'age': 'Age'}
            selected_demo_danger = st.selectbox(
                "Cross-tab by:",
                options=list(demo_options.keys()),
                format_func=lambda x: demo_options[x],
                key='danger_demo'
            )
        
        if selected_danger in filtered_df.columns:
            # Exclude PNA and Non-binary from cross-tabs
            exclude_values = ['Prefer not to answer']
            if selected_demo_danger == 'gender':
                exclude_values.append('Non-binary/Other')
            
            data = []
            for group in filtered_df[selected_demo_danger].dropna().unique():
                if group in exclude_values:
                    continue
                subset = filtered_df[filtered_df[selected_demo_danger] == group]
                dangerous = subset[selected_danger].isin(['Somewhat dangerous', 'Very dangerous']).sum()
                valid = subset[selected_danger].notna().sum() - (subset[selected_danger] == 'Prefer not to answer').sum()
                if valid > 0:
                    data.append({'Group': str(group), 'Pct': (dangerous / valid) * 100, 'N': valid})
            
            if data:
                chart_df = pd.DataFrame(data)
                total_n = chart_df['N'].sum()
                fig = px.bar(
                    chart_df, x='Group', y='Pct',
                    text=chart_df['Pct'].apply(lambda x: f'{x:.1f}%'),
                    color_discrete_sequence=[COLORS['warm']]
                )
                fig.update_traces(textposition='outside', textfont_color='#000000')
                fig.update_layout(
                    title={'text': f'% Rating as Dangerous: {DANGER_VARS[selected_danger]} (N={total_n})', 'font': {'size': 16, 'color': '#666666'}},
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                    font={'color': '#666666', 'size': 13},
                    xaxis_title=demo_options[selected_demo_danger],
                    yaxis_title='% rating as dangerous',
                    xaxis=dict(tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, showgrid=True, gridcolor='#e2e8f0'),
                    yaxis=dict(range=[0, 100], tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, showgrid=True, gridcolor='#e2e8f0'),
                    height=350,
                    margin=dict(l=20, r=20, t=50, b=20),
                    shapes=[{'type': 'rect', 'xref': 'paper', 'yref': 'paper', 'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1, 'line': {'color': '#e2e8f0', 'width': 1}}]
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # =========================================================================
    # TAB 4: ENFORCEMENT PERCEPTIONS
    # =========================================================================
    with tab4:
        st.header("Enforcement Risk Perceptions")
        st.markdown(f"**{len(filtered_df)} respondents** | *How likely is it you'd be pulled over while...*")
        
        fig_legal = create_stacked_perception_chart(
            filtered_df, LEGAL_VARS,
            'Perceived Likelihood of Being Pulled Over',
            LIKELIHOOD_ORDER, COLORS['diverging']
        )
        if fig_legal:
            st.plotly_chart(fig_legal, use_container_width=True)
        
        st.subheader("Cross-tabulation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_legal = st.selectbox(
                "Select item:",
                options=list(LEGAL_VARS.keys()),
                format_func=lambda x: LEGAL_VARS[x],
                key='legal_select'
            )
        
        with col2:
            demo_options = {'gender': 'Gender', 'city': 'Location', 'enroll': 'Enrollment', 'age': 'Age'}
            selected_demo_legal = st.selectbox(
                "Cross-tab by:",
                options=list(demo_options.keys()),
                format_func=lambda x: demo_options[x],
                key='legal_demo'
            )
        
        if selected_legal in filtered_df.columns:
            # Exclude PNA and Non-binary from cross-tabs
            exclude_values = ['Prefer not to answer']
            if selected_demo_legal == 'gender':
                exclude_values.append('Non-binary/Other')
            
            data = []
            for group in filtered_df[selected_demo_legal].dropna().unique():
                if group in exclude_values:
                    continue
                subset = filtered_df[filtered_df[selected_demo_legal] == group]
                likely = subset[selected_legal].isin(['Somewhat likely', 'Very likely']).sum()
                valid = subset[selected_legal].notna().sum() - (subset[selected_legal] == 'Prefer not to answer').sum()
                if valid > 0:
                    data.append({'Group': str(group), 'Pct': (likely / valid) * 100, 'N': valid})
            
            if data:
                chart_df = pd.DataFrame(data)
                total_n = chart_df['N'].sum()
                fig = px.bar(
                    chart_df, x='Group', y='Pct',
                    text=chart_df['Pct'].apply(lambda x: f'{x:.1f}%'),
                    color_discrete_sequence=[COLORS['primary']]
                )
                fig.update_traces(textposition='outside', textfont_color='#000000')
                fig.update_layout(
                    title={'text': f'% Rating as Likely: {LEGAL_VARS[selected_legal]} (N={total_n})', 'font': {'size': 16, 'color': '#666666'}},
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                    font={'color': '#666666', 'size': 13},
                    xaxis_title=demo_options[selected_demo_legal],
                    yaxis_title='% rating as likely',
                    xaxis=dict(tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, showgrid=True, gridcolor='#e2e8f0'),
                    yaxis=dict(range=[0, 100], tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, showgrid=True, gridcolor='#e2e8f0'),
                    height=350,
                    margin=dict(l=20, r=20, t=50, b=20),
                    shapes=[{'type': 'rect', 'xref': 'paper', 'yref': 'paper', 'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1, 'line': {'color': '#e2e8f0', 'width': 1}}]
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # =========================================================================
    # TAB 5: PEER NORMS
    # =========================================================================
    with tab5:
        st.header("Peer Behavior Perceptions")
        st.markdown(f"**{len(filtered_df)} respondents** | *How often do you think people your age...*")
        
        fig_norms = create_prevalence_chart(
            filtered_df, NORMS_VARS,
            '% Believing Peers Engage in Behavior (At Least Once)',
            color=COLORS['neutral']
        )
        if fig_norms:
            st.plotly_chart(fig_norms, use_container_width=True)
        
        st.subheader("Cross-tabulation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_norm = st.selectbox(
                "Select item:",
                options=list(NORMS_VARS.keys()),
                format_func=lambda x: NORMS_VARS[x],
                key='norm_select'
            )
        
        with col2:
            demo_options = {'gender': 'Gender', 'city': 'Location', 'enroll': 'Enrollment', 'age': 'Age'}
            selected_demo_norm = st.selectbox(
                "Cross-tab by:",
                options=list(demo_options.keys()),
                format_func=lambda x: demo_options[x],
                key='norm_demo'
            )
        
        fig_norm_cross = create_crosstab_chart(
            filtered_df, selected_norm, selected_demo_norm,
            demo_options[selected_demo_norm], NORMS_VARS[selected_norm]
        )
        if fig_norm_cross:
            st.plotly_chart(fig_norm_cross, use_container_width=True)
    
    # =========================================================================
    # TAB 6: DUI AVOIDANCE
    # =========================================================================
    with tab6:
        st.header("DUI Avoidance Methods")
        st.markdown(f"**{len(filtered_df)} respondents** | *Methods used to avoid driving impaired*")
        
        fig_avoid = create_prevalence_chart(
            filtered_df, AVOIDANCE_VARS,
            '% Using Each Method (At Least Once)',
            color=COLORS['accent']
        )
        if fig_avoid:
            st.plotly_chart(fig_avoid, use_container_width=True)
        
        st.subheader("Cross-tabulation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_avoid = st.selectbox(
                "Select method:",
                options=list(AVOIDANCE_VARS.keys()),
                format_func=lambda x: AVOIDANCE_VARS[x],
                key='avoid_select'
            )
        
        with col2:
            demo_options = {'gender': 'Gender', 'city': 'Location', 'enroll': 'Enrollment', 'age': 'Age'}
            selected_demo_avoid = st.selectbox(
                "Cross-tab by:",
                options=list(demo_options.keys()),
                format_func=lambda x: demo_options[x],
                key='avoid_demo'
            )
        
        fig_avoid_cross = create_crosstab_chart(
            filtered_df, selected_avoid, selected_demo_avoid,
            demo_options[selected_demo_avoid], AVOIDANCE_VARS[selected_avoid]
        )
        if fig_avoid_cross:
            st.plotly_chart(fig_avoid_cross, use_container_width=True)
    
    # =========================================================================
    # TAB 7: MICROMOBILITY
    # =========================================================================
    with tab7:
        st.header("Micromobility")
        st.markdown("*Overall prevalence only — small samples prevent demographic cross-tabs*")
        
        bikers = filtered_df[filtered_df['bike'].notna()] if 'bike' in filtered_df.columns else pd.DataFrame()
        scooters = filtered_df[filtered_df['scoot'].notna()] if 'scoot' in filtered_df.columns else pd.DataFrame()
        skaters = filtered_df[filtered_df['skate'].notna()] if 'skate' in filtered_df.columns else pd.DataFrame()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Bicyclists", len(bikers))
        col2.metric("Scooter Riders", len(scooters))
        col3.metric("Skateboarders", len(skaters))
        
        st.markdown("---")
        
        # BIKING
        if len(bikers) > 0:
            st.subheader("Bicycling")
            
            col1, col2 = st.columns(2)
            with col1:
                fig = create_distribution_chart(bikers, 'bikehelmet', 'Helmet Use', HELMET_ORDER, COLORS['categorical'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = create_distribution_chart(bikers, 'electricbike', 'Electric Bike Use', ELECTRIC_ORDER, COLORS['categorical'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            bike_vars = {'bikealc': '2 hrs after 3+ drinks', 'bikecann5': '5 hrs after cannabis', 'bikecann9': '9 hrs after cannabis', 'bikesim': '2 hrs after both'}
            fig = create_prevalence_chart(bikers, bike_vars, 'Impaired Biking Prevalence', COLORS['primary'])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # SCOOTERING
        if len(scooters) > 0:
            st.subheader("Scootering")
            
            col1, col2 = st.columns(2)
            with col1:
                fig = create_distribution_chart(scooters, 'scoothelmet', 'Helmet Use', HELMET_ORDER, COLORS['categorical'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = create_distribution_chart(scooters, 'escoot', 'Electric Scooter Use', ELECTRIC_ORDER, COLORS['categorical'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            scoot_vars = {'scootalc': '2 hrs after 3+ drinks', 'scootcann5': '5 hrs after cannabis', 'scootcann9': '9 hrs after cannabis', 'scootsim': '2 hrs after both'}
            fig = create_prevalence_chart(scooters, scoot_vars, 'Impaired Scootering Prevalence', COLORS['gold'])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # SKATEBOARDING
        if len(skaters) > 0:
            st.subheader("Skateboarding")
            
            col1, col2 = st.columns(2)
            with col1:
                fig = create_distribution_chart(skaters, 'skatehelmet', 'Helmet Use', HELMET_ORDER, COLORS['categorical'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = create_distribution_chart(skaters, 'electricskate', 'Electric Skateboard Use', ELECTRIC_ORDER, COLORS['categorical'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            skate_vars = {'skatealc': '2 hrs after 3+ drinks', 'skatecann5': '5 hrs after cannabis', 'skatecann9': '9 hrs after cannabis', 'skatesim': '2 hrs after both'}
            fig = create_prevalence_chart(skaters, skate_vars, 'Impaired Skateboarding Prevalence', COLORS['neutral'])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    # =========================================================================
    # TAB 8: BEHAVIOR COMPARISON
    # =========================================================================
    with tab8:
        st.header("Behavior Comparison")
        st.markdown(f"**{len(filtered_df)} respondents** | *Compare aggregate measures across behaviors*")
        st.markdown("Select a behavior to see its prevalence, danger perception, enforcement perception, and peer norms side-by-side.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Behavior 1")
            selected_behavior_1 = st.selectbox(
                "Select behavior:",
                options=list(BEHAVIOR_COMPARISON_VARS.keys()),
                format_func=lambda x: BEHAVIOR_COMPARISON_VARS[x]['label'],
                key='compare_behavior_1'
            )
            
            fig1 = create_behavior_comparison_chart(filtered_df, selected_behavior_1, BEHAVIOR_COMPARISON_VARS)
            if fig1:
                st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("Behavior 2")
            selected_behavior_2 = st.selectbox(
                "Select behavior:",
                options=list(BEHAVIOR_COMPARISON_VARS.keys()),
                format_func=lambda x: BEHAVIOR_COMPARISON_VARS[x]['label'],
                key='compare_behavior_2',
                index=1  # Default to second option
            )
            
            fig2 = create_behavior_comparison_chart(filtered_df, selected_behavior_2, BEHAVIOR_COMPARISON_VARS)
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("---")
        st.markdown("""
        **Measure definitions:**
        - **Prevalence**: % who engaged in the behavior at least once (past 30 days)
        - **Rated Dangerous**: % who rated the behavior as "Somewhat dangerous" or "Very dangerous"
        - **Rated Likely (Enforcement)**: % who rated being pulled over as "Somewhat likely" or "Very likely"
        - **Peers Engage**: % who believe their peers engage in the behavior at least once
        """)

if __name__ == "__main__":
    main()
