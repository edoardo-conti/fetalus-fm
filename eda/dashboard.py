import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Load the CSV data
df = pd.read_csv('eda/datasets_eda_metrics.csv')

# Initialize Dash app with minimal theme
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

app.layout = html.Div([
    # Title Section
    html.Div([
        html.H1("Fetal Datasets", style={'textAlign': 'center'})
    ], className='row', style={'marginBottom': '20px'}),
    
    # Main Tabs
    dcc.Tabs(id='main-tabs', value='aggregated-datasets', children=[
        dcc.Tab(label='Single Dataset', children=[
            html.Div(className='row', children=[
                html.Div(className='twelve columns', children=[
                    dcc.Dropdown(
                        id='dataset-selector',
                        options=[{'label': ds, 'value': ds} for ds in df['DATASET']],
                        value=df['DATASET'][0],
                        clearable=False,
                        style={'marginBottom': '20px'}
                    )
                ])
            ]),
            
            html.Div(className='row', children=[
                html.Div(className='twelve columns', children=[
                    html.Div(id='metrics-display', style={
                        'padding': '20px', 
                        'border': '1px solid #eee', 
                        'borderRadius': '5px',
                        'marginBottom': '20px'
                    })
                ])
            ]),
    
            html.Div(className='row', children=[
                html.Div(className='six columns', children=[
                    dcc.Graph(id='patient-split', style={'height': '500px', 'marginBottom': '20px'})
                ]),
                html.Div(className='six columns', children=[
                    dcc.Graph(id='image-split', style={'height': '500px', 'marginBottom': '20px'})
                ])
            ]),
            
            html.Div(className='row', children=[
                html.Div(className='six columns', children=[
                    dcc.Graph(id='mask-split', style={'height': '500px', 'marginBottom': '20px'})
                ]),
                html.Div(className='six columns', children=[
                    dcc.Graph(id='class-distribution', style={'height': '500px', 'marginBottom': '20px'}
                    )
                ])
            ]),
            
            html.Div(className='row', children=[
                html.Div(className='twelve columns', children=[
                    dcc.Graph(
                        id='structure-distribution',
                        style={'height': '550px', 'marginBottom': '20px'}
                    )
                ])
            ]),
        ]),
        
        dcc.Tab(label='Aggregated Datasets', value='aggregated-datasets', children=[
            html.Div(className='row', children=[
                html.Div(className='twelve columns', children=[
                    html.Div(style={'padding': '20px', 'border': '1px solid #eee', 'borderRadius': '5px'}, children=[
                        html.H3("Aggregated Metrics"),
                        html.P(f"Datasets: {len(df)}"),
                        html.P(f"Total patients: {df['NUM_PATIENTS'].sum()}"),
                        html.Ul([
                            html.Li(f"Train patients: {df['NUM_PATIENTS_TRAIN'].sum()}"),
                            html.Li(f"Test patients: {df['NUM_PATIENTS_TEST'].sum()}")
                        ]),
                        html.P(f"Total images: {df['NUM_IMGS'].sum()}"),
                        html.Ul([
                            html.Li(f"Train images: {df['NUM_IMGS_TRAIN'].sum()}"),
                            html.Li(f"Test images: {df['NUM_IMGS_TEST'].sum()}"),
                        ]),
                        html.P(f"Total masks: {df['NUM_MASKS'].sum()}"),
                        html.Ul([
                            html.Li(f"Train sets masks: {df['NUM_MASKS_TRAIN'].sum()}"),
                            html.Li(f"Test sets masks: {df[df['TEST_ANNOTATED'] == 'AVAILABLE']['NUM_MASKS_TEST'].sum()}"),
                        ]),
                        html.P(
                            f"Dataset with test annotations: {', '.join(s[1:] for s in (df[df['TEST_ANNOTATED'] == 'AVAILABLE']['DATASET'].values))} "
                            f"({sum(df['TEST_ANNOTATED'] == 'AVAILABLE')}/{len(df)})"
                        )
                    ])
                ])
            ]),
            
            html.Div(className='row', children=[
                html.Div(className='four columns', children=[
                    dcc.Graph(
                        id='aggregated-patient-split',
                        figure=px.pie(
                            names=['Train', 'Test'],
                            values=[df['NUM_PATIENTS_TRAIN'].sum(), df['NUM_PATIENTS_TEST'].sum()],
                            title='Aggregated Patient Split',
                            hole=0.3
                        ).update_traces(
                            textinfo='label+percent+value',
                            textposition='outside',
                            textfont=dict(size=18)
                        ).update_layout(
                            uniformtext_minsize=18,
                            uniformtext_mode='hide'
                        )
                    )
                ]),
                html.Div(className='four columns', children=[
                    dcc.Graph(
                        id='aggregated-image-split',
                        figure=px.pie(
                            names=['Train', 'Test'],
                            values=[df['NUM_IMGS_TRAIN'].sum(), df['NUM_IMGS_TEST'].sum()],
                            title='Aggregated Image Split',
                            hole=0.3
                        ).update_traces(
                            textinfo='label+percent+value',
                            textposition='outside',
                            textfont=dict(size=18),
                            rotation=-30
                        ).update_layout(
                            uniformtext_minsize=18,
                            uniformtext_mode='hide',
                            #margin=dict(t=100)  # Aumenta margine superiore
                        )
                    )
                ]),
                html.Div(className='four columns', children=[
                    dcc.Graph(
                        id='aggregated-mask-split',
                        figure=px.pie(
                            names=['Train', 'Test'],
                            values=[df['NUM_MASKS_TRAIN'].sum(), df[df['TEST_ANNOTATED'] == 'AVAILABLE']['NUM_MASKS_TEST'].sum()],
                            title='Aggregated Mask Split',
                            hole=0.3
                        ).update_traces(
                            textinfo='label+percent+value',
                            textposition='outside',
                            textfont=dict(size=18),
                            rotation=-10
                        ).update_layout(
                            uniformtext_minsize=18,
                            uniformtext_mode='hide'
                        )
                    )
                ])
            ]),
            
            html.Div(className='row', children=[
                html.Div(className='twelve columns', children=[
                    dcc.Checklist(
                        id='log-scale-toggle',
                        options=[{'label': 'Use logarithmic scale', 'value': 'log'}],
                        value=['log'],
                        style={'marginBottom': '10px'}
                    ),
                    dcc.Graph(
                        id='aggregated-classes', 
                        style={'height': '500px', 'marginBottom': '20px'}
                    )
                ])
            ]),
            
            html.Div(className='row', children=[
                html.Div(className='twelve columns', children=[
                    dcc.Graph(
                        id='aggregated-structures',
                        style={'height': '500px', 'marginBottom': '20px'}
                    )
                ])
            ]),

            html.Div(className='row', children=[
                html.Div(className='twelve columns', children=[
                    dcc.Graph(
                        id='mask-annotations-per-structure',
                        style={'height': '650px', 'marginBottom': '20px'}
                    )
                ])
            ]),
            
            html.Div(className='row', children=[
                html.Div(className='six columns', children=[
                    dcc.Graph(
                        id='test-availability',
                        style={'height': '450px'}
                    )
                ]),
                html.Div(className='six columns', children=[
                    dcc.Graph(
                        id='split-methods',
                        style={'height': '450px'}
                    )
                ])
            ])
        ])
        ])
    ])

@app.callback(
    [Output('metrics-display', 'children'),
     Output('patient-split', 'figure'),
     Output('image-split', 'figure'),
     Output('mask-split', 'figure'), 
     Output('class-distribution', 'figure'),
     Output('structure-distribution', 'figure'),
     Output('aggregated-classes', 'figure'),
     Output('aggregated-structures', 'figure'),
     Output('mask-annotations-per-structure', 'figure'),
     Output('test-availability', 'figure'),
     Output('split-methods', 'figure')],
    [Input('dataset-selector', 'value'),
     Input('log-scale-toggle', 'value')]
)
def update_dashboard(selected_dataset, log_scale):
    dataset = df[df['DATASET'] == selected_dataset].iloc[0]
    
    # Calculate percentages
    train_mask_perc = dataset['NUM_MASKS_TRAIN'] / dataset['NUM_MASKS'] * 100 if dataset['NUM_MASKS'] > 0 else 0
    test_mask_perc = dataset['NUM_MASKS_TEST'] / dataset['NUM_MASKS'] * 100 if dataset['NUM_MASKS'] > 0 else 0
    opt_mask_on_tot = dataset['NUM_MASKS_OPT'] / dataset['NUM_MASKS'] * 100 if dataset['NUM_MASKS_OPT'] > 0 else 0
    subopt_mask_on_tot = dataset['NUM_MASKS_SUBOPT'] / dataset['NUM_MASKS'] * 100 if dataset['NUM_MASKS_SUBOPT'] > 0 else 0

    # Basic metrics display
    metrics = [
        html.H3(f"Dataset: {selected_dataset}"),
        html.P(f"Classes: {dataset['CLASSES']} ({dataset['NUM_CLASSES']})"),
        html.Ul([
            html.Li(f"Patients: {dataset['NUM_PATIENTS']}"),
            html.Ul([
                html.Li(f"Train patients: {dataset['NUM_PATIENTS_TRAIN']}"),
                html.Li(f"Test patients: {dataset['NUM_PATIENTS_TEST']}")
            ])
        ]) if dataset['NUM_PATIENTS'] > 0 else '',
        html.P(f"Number of videos: {int(dataset['NUM_VIDEO'])}") if pd.notna(dataset['NUM_VIDEO']) else '',
        html.Ul([
            html.Li(f"Images: {dataset['NUM_IMGS']}", style={'listStyle': 'none'}),
            html.Ul([
                html.Li(f"Train images: {dataset['NUM_IMGS_TRAIN']} ({dataset['PERC_IMGS_TRAIN']:.1f}%)"),
                html.Li(f"Test images: {dataset['NUM_IMGS_TEST']} ({dataset['PERC_IMGS_TEST']:.1f}%)")
            ])
        ]),
        html.Ul([
            html.Li(f"Masks: {dataset['NUM_MASKS']} ({dataset['PERC_MASKS']:.1f}% of the images)", style={'listStyle': 'none'}),
            html.Ul([
                html.Li(f"Optimal masks: {int(dataset['NUM_MASKS_OPT'])} ({opt_mask_on_tot:.1f}%)") if selected_dataset == 'facouslic' else '',
                html.Li(f"Suboptimal masks: {int(dataset['NUM_MASKS_SUBOPT'])} ({subopt_mask_on_tot:.1f}%)") if selected_dataset == 'facouslic' else '',
                html.Li(f"Train masks: {dataset['NUM_MASKS_TRAIN']} ({train_mask_perc:.1f}%)"),
                html.Li(f"Test masks: {dataset['NUM_MASKS_TEST']} ({test_mask_perc:.1f}%)")
            ]),
        ]),
        html.P(f"Structures: {dataset['STRUCTURES'] if pd.notna(dataset['STRUCTURES']) else 'N/A'} ({dataset['NUM_STRUCTURES']})"),
        html.P(f"Split method: {dataset['SPLIT']}"),
        html.P(f"Test annotated: {dataset['TEST_ANNOTATED']}")
    ]
    
    # Patient split visualization
    if dataset['NUM_PATIENTS'] > 0:
        patient_fig = px.pie(
            names=['Train', 'Test'],
            values=[dataset['NUM_PATIENTS_TRAIN'], dataset['NUM_PATIENTS_TEST']],
            title=f"Patient Split - {selected_dataset}",
            hole=0.3,
            labels={'label': 'Split', 'value': 'Count'}
        )
        patient_fig.update_traces(
            textinfo='label+percent+value',
            textposition='outside',
            insidetextorientation='radial',
            texttemplate='%{label}<br>%{percent:.1%}<br>(%{value})',
            hovertemplate='%{label}: %{percent:.1%} (%{value})<extra></extra>',
            textfont=dict(size=18)
        )
        patient_fig.update_layout(
            uniformtext_minsize=18,
            uniformtext_mode='hide',
            font=dict(size=18)
        )
    else:
        patient_fig = px.pie(title="No patient data available")
    
    # Image split visualization
    img_fig = px.pie(
        names=['Train', 'Test'],
        values=[dataset['NUM_IMGS_TRAIN'], dataset['NUM_IMGS_TEST']],
        title=f"Image Split - {selected_dataset}",
        hole=0.3,
        labels={'label': 'Split', 'value': 'Count'}
    )
    img_fig.update_traces(
        textinfo='label+percent+value',
        textposition='outside',
        insidetextorientation='radial',
        texttemplate='%{label}<br>%{percent:.1%}<br>(%{value})',
        hovertemplate='%{label}: %{percent:.1%} (%{value})<extra></extra>',
        textfont=dict(size=18)
    )
    img_fig.update_layout(
        uniformtext_minsize=18,
        uniformtext_mode='hide',
        font=dict(size=18)
    )
        
    # Mask split visualization
    if dataset['NUM_MASKS'] > 0:
        mask_split_fig = px.pie(
            names=['Train', 'Test'],
            values=[dataset['NUM_MASKS_TRAIN'], dataset['NUM_MASKS_TEST']],
            title=f"Mask Split - {selected_dataset}",
            hole=0.3,
            labels={'label': 'Split', 'value': 'Count'}
        )
        mask_split_fig.update_traces(
            textinfo='label+percent+value',
            textposition='outside',
            insidetextorientation='radial',
            texttemplate='%{label}<br>%{percent:.1%}<br>(%{value})',
            hovertemplate='%{label}: %{percent:.1%} (%{value})<extra></extra>',
            textfont=dict(size=18)
        )
        mask_split_fig.update_layout(
            uniformtext_minsize=18,
            uniformtext_mode='hide',
            font=dict(size=18)
        )
    else:
        mask_split_fig = px.pie(title="No masks in this dataset")

    # Class distribution
    if pd.notna(dataset['DISTR_CLASSES']):
        classes = dataset['DISTR_CLASSES'].split('|')
        class_names = [c.split(':')[0] for c in classes]
        class_counts = [int(c.split(':')[1]) for c in classes]
        total = sum(class_counts)
        class_percents = [round(count/total*100, 1) for count in class_counts]
        #text = [f"{name}<br>{percent}% ({count})" for name, count, percent in zip(class_names, class_counts, class_percents)]
        
        class_fig = px.pie(
            names=class_names,
            values=class_counts,
            title=f"Class Distribution - {selected_dataset}",
            hole=0.3
        )
        class_fig.update_traces(
            #text=text,
            textinfo='label+percent+value',
            texttemplate='%{label}<br>%{percent:.1%}<br>(%{value})',
            textposition='outside',
            textfont=dict(size=18),
            hovertemplate='%{label}: %{percent:.1%} (%{value})<extra></extra>'
        )
        class_fig.update_layout(
            uniformtext_minsize=18,
            uniformtext_mode='hide',
            font=dict(size=18)
        )
    else:
        class_fig = px.pie(title="No class distribution data")
    
    # Structure distribution histogram
    struct_hist_fig = px.bar(title="No structure distribution data")
    if pd.notna(dataset['DISTR_STRUCTURES']):
        structures_data = []
        for item in dataset['DISTR_STRUCTURES'].split('|'):
            if ':' in item:
                struct, count = item.split(':')
                if '-' in struct:
                    # Dividi le strutture combinate e mantieni lo stesso conteggio per ciascuna
                    for s in struct.split('-'):
                        structures_data.append({
                            'Structure': s,
                            'Count': int(count)
                        })
                else:
                    structures_data.append({
                        'Structure': struct,
                        'Count': int(count)
                    })
        
        if structures_data:
            # Crea DataFrame e aggrega i conteggi per struttura
            struct_df = pd.DataFrame(structures_data)
            struct_df = struct_df.groupby('Structure')['Count'].sum().reset_index()
            struct_df = struct_df.sort_values('Count', ascending=False)
            
            struct_hist_fig = px.bar(
                struct_df,
                x='Structure',
                y='Count',
                title=f"Structure Distribution in Masks - {selected_dataset}",
                labels={'Structure': 'Structure', 'Count': 'Number of Masks'},
                text='Count'
            )
            struct_hist_fig.update_traces(
                textposition='outside',
                textfont=dict(size=18),
                marker_line_color='rgb(8,48,107)',
                marker_line_width=1.5
            )
            struct_hist_fig.update_layout(
                font=dict(size=18),
                height=500,
                margin=dict(pad=10),
                autosize=True,
                xaxis=dict(
                    automargin=True,
                )
            )

    # Mask split visualization
    if dataset['NUM_MASKS'] > 0:
        mask_fig = px.pie(
            names=['Train', 'Test'],
            values=[dataset['NUM_MASKS_TRAIN'], dataset['NUM_MASKS_TEST']],
            title=f"Mask Split - {selected_dataset}",
            hole=0.3,
            labels={'label': 'Split', 'value': 'Count'}
        )
        mask_fig.update_traces(
            textinfo='label+percent+value',
            textposition='outside',
            insidetextorientation='radial',
            texttemplate='%{label}<br>%{percent:.1%}<br>(%{value})',
            hovertemplate='%{label}: %{percent:.1%} (%{value})<extra></extra>',
            textfont=dict(size=18)
        )
        mask_fig.update_layout(
            uniformtext_minsize=18,
            uniformtext_mode='hide',
            font=dict(size=18)
        )
    else:
        mask_fig = px.pie(title="No masks in this dataset")
    
    # Structure distribution
    if pd.notna(dataset['STRUCTURES']) and dataset['NUM_STRUCTURES'] > 0:
        structures = []
        counts = []
        
        # Parse structure distribution
        if pd.notna(dataset['DISTR_STRUCTURES']):
            # Split combined structures and aggregate their counts
            struct_dict = {}  # Dictionary to store structure:count pairs
            for distr in dataset['DISTR_STRUCTURES'].split('|'):
                if ':' in distr:
                    struct, count = distr.split(':')
                    count = int(count)
                    if '-' in struct:
                        # Handle combined structures (e.g. "artery-liver-stomach-vein")
                        for s in struct.split('-'):
                            struct_dict[s] = struct_dict.get(s, 0) + count
                    else:
                        struct_dict[struct] = struct_dict.get(struct, 0) + count
            
            # Convert dictionary to lists for plotting
            structures = list(struct_dict.keys())
            counts = list(struct_dict.values())
        
        if structures:
            struct_df = pd.DataFrame({'Structure': structures, 'Count': counts})
            struct_df = struct_df.sort_values('Count', ascending=True)  # Sort by count
            
            struct_fig = px.bar(
                struct_df,
                x='Structure',
                y='Count',
                title=f"Structure Distribution - {selected_dataset}",
                labels={'Structure': 'Structure', 'Count': 'Count'},
                text='Count',
                color='Structure',
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            struct_fig.update_traces(
                textposition='outside',
                textfont_size=18,
                marker_line_color='rgb(8,48,107)',
                marker_line_width=1.5
            )
            struct_fig.update_layout(showlegend=False)
        else:
            struct_fig = px.bar(title="No structure distribution data")
    else:
        struct_fig = px.bar(title="No structure data")
    
    # Special facouslic view
    facouslic_style = {'display': 'block'} if selected_dataset == 'facouslic' else {'display': 'none'}
    
    # Prepare aggregated class data
    class_data = []
    for _, row in df.iterrows():
        if pd.notna(row['DISTR_CLASSES']):
            for item in row['DISTR_CLASSES'].split('|'):
                if ':' in item:
                    class_name, count = item.split(':')
                    class_data.append({
                        'Dataset': row['DATASET'],
                        'Class': class_name,
                        'Count': int(count)
                    })
    
    if class_data:
        class_df = pd.DataFrame(class_data)
        agg_class_fig = px.bar(
            class_df,
            x='Class',
            y='Count',
            color='Dataset',
            title='Class Distribution Across All Datasets',
            barmode='stack'
        )
        if 'log' in log_scale:
            agg_class_fig.update_yaxes(type='log')
    else:
        agg_class_fig = px.bar(title="No class distribution data available")

    # Prepare aggregated structure data
    struct_data = []
    temp_dict = {}  # Temporary dictionary to aggregate counts
    
    for _, row in df.iterrows():
        if pd.notna(row['DISTR_STRUCTURES']):
            dataset = row['DATASET']
            # Initialize dataset in temp_dict if not exists
            if dataset not in temp_dict:
                temp_dict[dataset] = {}
                
            for distr in row['DISTR_STRUCTURES'].split('|'):
                if ':' in distr:
                    struct, count = distr.split(':')
                    count = int(count)
                    if '-' in struct:
                        for s in struct.split('-'):
                            # Add count to existing value or initialize
                            temp_dict[dataset][s] = temp_dict[dataset].get(s, 0) + count
                    else:
                        temp_dict[dataset][struct] = temp_dict[dataset].get(struct, 0) + count
    
    # Convert aggregated dictionary to list of dictionaries
    for dataset, structures in temp_dict.items():
        for struct, count in structures.items():
            struct_data.append({
                'Dataset': dataset,
                'Structure': struct,
                'Count': count
            })

    if struct_data:
        struct_df = pd.DataFrame(struct_data)
        agg_struct_fig = px.bar(
            struct_df,
            x='Structure',
            y='Count',
            color='Dataset',
            title='Structure Distribution Across All Datasets',
            barmode='stack'
        )
        if 'log' in log_scale:
            agg_struct_fig.update_yaxes(type='log')
    else:
        agg_struct_fig = px.bar(title="No structure distribution data available")
    
    # Test availability pie chart with dataset lists
    test_annotated = df.groupby('TEST_ANNOTATED')['DATASET'].agg(['count', lambda x: ', '.join(x)]).reset_index()
    test_annotated.columns = ['Status', 'Count', 'Datasets']
    test_fig = px.pie(
        test_annotated,
        names='Status',
        values='Count',
        title='Test Set Annotation Availability',
        hole=0.3,
        custom_data=['Datasets']
    )
    test_fig.update_traces(
        textposition='outside',
        textinfo='label+value',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Datasets: %{customdata}<extra></extra>',
        texttemplate='%{label}<br>%{value}<br>[%{customdata}]<br>%{percent:.1%}',
        textfont=dict(size=14)
    )
    test_fig.update_layout(
        uniformtext_minsize=12,
        uniformtext_mode='hide',
        font=dict(size=14)
    )
    
    # Split methods pie chart
    split_methods = df['SPLIT'].value_counts().reset_index()
    split_methods.columns = ['Method', 'Count']
    split_fig = px.pie(
        split_methods,
        names='Method',
        values='Count',
        title='Dataset Split Methods',
        hole=0.3
    )
    split_fig.update_traces(
        textposition='outside',
        textinfo='label+percent+value',
        texttemplate='%{label}<br>%{value}<br>%{percent}',
        hovertemplate='%{label}: %{percent:.1f}% (%{value})<extra></extra>',
        textfont=dict(size=18),
        rotation=-20
    )
    split_fig.update_layout(
        uniformtext_minsize=18,
        uniformtext_mode='hide',
        showlegend=True
    )
    
    # Prepare mask annotations per structure data
    annotations_data = []
    for _, row in df.iterrows():
        if pd.notna(row['DISTR_STRUCTURES']):
            for item in row['DISTR_STRUCTURES'].split('|'):
                if ':' in item:
                    structure, count = item.split(':')
                    # Handle combined structures (e.g. "artery-liver-stomach-vein")
                    if '-' in structure:
                        for s in structure.split('-'):
                            annotations_data.append({
                                'Dataset': row['DATASET'],
                                'Structure': s,
                                'Mask Count': int(count)
                            })
                    else:
                        annotations_data.append({
                            'Dataset': row['DATASET'],
                            'Structure': structure,
                            'Mask Count': int(count)
                        })
    
    if annotations_data:
        annotations_df = pd.DataFrame(annotations_data)
        # Aggregate counts for same structure-dataset combinations
        annotations_df = annotations_df.groupby(['Dataset', 'Structure'])['Mask Count'].sum().reset_index()
        
        annotations_fig = px.bar(
            annotations_df,
            y='Structure',
            x='Mask Count',
            color='Dataset',
            title='Mask Annotations per Structure',
            barmode='group',
            text='Mask Count',
            orientation='h'
        )
        annotations_fig.update_traces(
            textfont_size=18,  # Aumenta dimensione testo barre
            textposition='outside',
            marker=dict(line=dict(width=1)),
            width=0.3  # Aumenta larghezza barre
        )
        annotations_fig.update_layout(
            yaxis_title='Structures',
            font=dict(size=18),  # Aumenta dimensione font generale
            bargap=0.0,  # Aumenta spazio tra barre
            height=650  # Aumenta altezza grafico
        )
        if 'log' in log_scale:
            annotations_fig.update_xaxes(type='log')
    else:
        annotations_fig = px.bar(title="No annotation data available")

    return (
        metrics,          # metrics-display
        patient_fig,      # patient-split
        img_fig,         # image-split
        mask_split_fig,  # mask-split
        class_fig,       # class-distribution
        struct_hist_fig, # structure-distribution
        agg_class_fig,   # aggregated-classes
        agg_struct_fig,  # aggregated-structures
        annotations_fig,  # mask-annotations-per-structure
        test_fig,        # test-availability
        split_fig,       # split-methods
    )

if __name__ == '__main__':
    app.run(debug=True)
