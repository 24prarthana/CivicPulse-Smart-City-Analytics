from pathlib import Path
from io import BytesIO
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, IsolationForest, RandomForestClassifier
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="CivicPulse | Smart City Intelligence", page_icon="🏙️", layout="wide")
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "smart_city_incidents.csv"

# ----------------------------- THEME -----------------------------------------
with st.sidebar:
    st.markdown("# 🏙️ CivicPulse")
    theme = st.radio("Appearance", ["Dark", "Light"], horizontal=True, index=0)
if theme == "Dark":
    bg, panel, card, fg, muted, border, accent = "#08111F", "#101C2E", "#15243A", "#EAF1FB", "#9BAEC5", "#273B55", "#51D6C2"
    template = "plotly_dark"
else:
    bg, panel, card, fg, muted, border, accent = "#F3F6FB", "#FFFFFF", "#FFFFFF", "#172337", "#5E6D82", "#DCE4EF", "#087F8C"
    template = "plotly_white"
st.markdown(f"""<style>
.stApp{{background:{bg};color:{fg}}}[data-testid="stHeader"]{{background:{bg}}}
[data-testid="stSidebar"]{{background:{panel};border-right:1px solid {border}}}
[data-testid="stSidebar"] *{{color:{fg}}} h1,h2,h3{{color:{fg}!important;letter-spacing:-.025em}}
p,label,.stCaption{{color:{muted}}}
[data-testid="stMetric"]{{background:linear-gradient(135deg,{card},{panel});border:1px solid {border};padding:17px;border-radius:15px;box-shadow:0 5px 18px rgba(0,0,0,.08)}}
[data-testid="stMetricLabel"]{{color:{muted}!important}}[data-testid="stMetricValue"]{{color:{fg}!important}}
.stButton button,.stDownloadButton button{{border-radius:10px;border:1px solid {accent}}}
.block-container{{padding-top:1.8rem;padding-bottom:3rem;max-width:1600px}}hr{{border-color:{border}}}
</style>""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    d=pd.read_csv(DATA_PATH,parse_dates=['Reported_At'])
    d['Month']=d.Reported_At.dt.to_period('M').astype(str)
    d['Weekday']=d.Reported_At.dt.day_name()
    d['Day_Type']=np.where(d.Reported_At.dt.dayofweek>=5,'Weekend','Weekday')
    d['SLA_Met']=np.where(d.Status.eq('Resolved'),d.Resolution_Hours<=24,np.nan)
    return d

df=load_data()
if 'page' not in st.session_state: st.session_state.page='🏠 Overview'
with st.sidebar:
    st.divider()
    page=st.radio('Navigate', ['🏠 Overview','📊 Analytics','🗺️ Incident Map','🤖 Predictions','📈 Performance','📄 Reports'], key='page')
    st.caption('Portfolio prototype • Synthetic demo data')
    st.divider(); st.subheader('Global filters')
    areas=st.multiselect('Areas',sorted(df.Area.unique()),default=sorted(df.Area.unique()),key='areas')
    types=st.multiselect('Incident types',sorted(df.Incident_Type.unique()),default=sorted(df.Incident_Type.unique()),key='types')
    severities=st.multiselect('Severity', ['Low','Medium','High','Critical'],default=['Low','Medium','High','Critical'],key='severities')
    statuses=st.multiselect('Status',sorted(df.Status.unique()),default=sorted(df.Status.unique()),key='statuses')
    dmin,dmax=df.Reported_At.min().date(),df.Reported_At.max().date()
    date_range=st.date_input('Reported date range',(dmin,dmax),min_value=dmin,max_value=dmax,key='date_range')
    with st.expander('Advanced filters'):
        res_values=df.Resolution_Hours.dropna()
        res_range=st.slider('Resolution hours',0.0,float(max(1,res_values.max())),(0.0,float(max(1,res_values.max()))),key='res_range')
        rate_values=df.Citizen_Rating.dropna()
        rating_range=st.slider('Citizen rating',1.0,5.0,(1.0,5.0),0.5,key='rating_range')
        cost_max=float(df.Estimated_Cost_INR.max())
        cost_range=st.slider('Estimated cost (₹)',0.0,cost_max,(0.0,cost_max),key='cost_range')
        day_type=st.selectbox('Day type',['All','Weekday','Weekend'],key='day_type')

if isinstance(date_range,(tuple,list)) and len(date_range)==2: start,end=date_range
else: start,end=dmin,dmax
filtered=df[df.Area.isin(areas)&df.Incident_Type.isin(types)&df.Severity.isin(severities)&df.Status.isin(statuses)&(df.Reported_At.dt.date>=start)&(df.Reported_At.dt.date<=end)].copy()
filtered=filtered[filtered.Resolution_Hours.fillna(-1).between(res_range[0],res_range[1]) | filtered.Resolution_Hours.isna()]
filtered=filtered[filtered.Citizen_Rating.fillna(-1).between(rating_range[0],rating_range[1]) | filtered.Citizen_Rating.isna()]
filtered=filtered[filtered.Estimated_Cost_INR.between(cost_range[0],cost_range[1])]
if day_type!='All': filtered=filtered[filtered.Day_Type.eq(day_type)]

# Models use the full demo dataset; user filters affect reporting and visuals.
def style_fig(fig):
    fig.update_layout(template=template,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font_color=fg,margin=dict(l=10,r=10,t=55,b=10))
    return fig

def insight_text(d):
    if d.empty: return 'No records match the selected filters. Adjust the filters to generate an executive summary.'
    area=d.Area.value_counts().idxmax(); area_n=int(d.Area.value_counts().max())
    typ=d.Incident_Type.value_counts().idxmax(); typ_n=int(d.Incident_Type.value_counts().max())
    critical=int(d.Severity.eq('Critical').sum())
    return f"**{area}** recorded the highest incident volume with **{area_n:,} reports**. **{typ}** was the most common incident type (**{typ_n:,} reports**). There are **{critical:,} critical incidents** in the selected dataset."

def make_excel(d):
    out=BytesIO()
    with pd.ExcelWriter(out,engine='openpyxl') as writer:
        d.to_excel(writer,index=False,sheet_name='Incidents')
        d.groupby('Area').agg(Incidents=('Incident_ID','count'),Estimated_Cost_INR=('Estimated_Cost_INR','sum'),Median_Resolution_Hours=('Resolution_Hours','median')).reset_index().to_excel(writer,index=False,sheet_name='Area Summary')
        d.groupby('Incident_Type').agg(Incidents=('Incident_ID','count'),Estimated_Cost_INR=('Estimated_Cost_INR','sum'),Median_Resolution_Hours=('Resolution_Hours','median')).reset_index().to_excel(writer,index=False,sheet_name='Type Summary')
    return out.getvalue()

def make_pdf(d):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    out=BytesIO(); doc=SimpleDocTemplate(out,pagesize=letter, rightMargin=35,leftMargin=35,topMargin=35,bottomMargin=35)
    styles=getSampleStyleSheet(); story=[Paragraph('CivicPulse — Incident Summary',styles['Title']),Spacer(1,10)]
    story.append(Paragraph(f"Reporting period: {start} to {end}",styles['Normal'])); story.append(Spacer(1,8))
    story.append(Paragraph(f"Total incidents: {len(d):,} | Critical: {int(d.Severity.eq('Critical').sum()):,} | Estimated cost: ₹{d.Estimated_Cost_INR.sum():,.0f}",styles['Normal'])); story.append(Spacer(1,12))
    story.append(Paragraph('Executive summary',styles['Heading2'])); story.append(Paragraph(insight_text(d).replace('**',''),styles['BodyText'])); story.append(Spacer(1,12))
    summary=d.groupby('Area').agg(Incidents=('Incident_ID','count'),Cost=('Estimated_Cost_INR','sum'),Median_Hours=('Resolution_Hours','median')).reset_index().sort_values('Incidents',ascending=False)
    table_data=[['Area','Incidents','Estimated cost (INR)','Median resolution (hrs)']]
    for _,r in summary.iterrows(): table_data.append([str(r.Area),str(int(r.Incidents)),f"₹{r.Cost:,.0f}",f"{r.Median_Hours:.1f}" if pd.notna(r.Median_Hours) else 'N/A'])
    table=Table(table_data,repeatRows=1,colWidths=[120,75,145,130]); table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#17324D')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),.4,colors.lightgrey),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F1F5F9')]),('PADDING',(0,0),(-1,-1),6)])); story.append(table)
    story.append(Spacer(1,10)); story.append(Paragraph('Note: synthetic portfolio demonstration data; not official city data.',styles['Italic']))
    doc.build(story); return out.getvalue()

st.title(page)
st.caption('SMART CITY INCIDENT INTELLIGENCE • PYTHON • STREAMLIT • MACHINE LEARNING')
st.info('Demonstration project using synthetic incidents and illustrative map coordinates. Not official city data or a real operational decision tool.')
if filtered.empty:
    st.warning('No records match the current filters. Change one or more filters in the sidebar.')
else:
    resolved=filtered[filtered.Status.eq('Resolved')]
    # ---------- OVERVIEW ----------
    if page=='🏠 Overview':
        st.subheader('Executive summary')
        st.markdown(insight_text(filtered))
        counts=filtered.Area.value_counts(); most_area=counts.index[0] if len(counts) else 'N/A'
        types_count=filtered.Incident_Type.value_counts(); common_type=types_count.index[0] if len(types_count) else 'N/A'
        critical=int(filtered.Severity.eq('Critical').sum())
        avg_rating=resolved.Citizen_Rating.mean()
        area_perf=resolved.groupby('Area').Resolution_Hours.median().dropna().sort_values()
        fastest=area_perf.index[0] if len(area_perf) else 'N/A'
        metrics=[('📍 Most affected area',most_area),('🧩 Common incident',common_type),('🚨 Critical incidents',f'{critical:,}'),('⭐ Avg citizen rating',f'{avg_rating:.2f}/5' if pd.notna(avg_rating) else 'N/A'),('⚡ Fastest responding area',fastest)]
        cols=st.columns(5)
        for c,(label,value) in zip(cols,metrics): c.metric(label,value)
        st.divider(); a,b=st.columns(2)
        with a:
            st.subheader('Incident trend')
            t=filtered.groupby('Month').size().reset_index(name='Incidents'); st.plotly_chart(style_fig(px.line(t,x='Month',y='Incidents',markers=True,title='Reports per month')),use_container_width=True)
        with b:
            st.subheader('Area hotspots')
            a_data=filtered.Area.value_counts().rename_axis('Area').reset_index(name='Incidents').sort_values('Incidents',ascending=False); st.plotly_chart(style_fig(px.bar(a_data,x='Area',y='Incidents',color='Incidents',title='Reports by area')),use_container_width=True)
        st.subheader('Latest incidents'); st.dataframe(filtered.sort_values('Reported_At',ascending=False).head(15),use_container_width=True,hide_index=True)
    # ---------- ANALYTICS ----------
    elif page=='📊 Analytics':
        st.subheader('Trend comparison and drill-down')
        c1,c2=st.columns(2)
        with c1: area_choice=st.multiselect('Compare areas',sorted(filtered.Area.unique()),default=sorted(filtered.Area.unique())[:2],key='compare_areas')
        with c2: type_choice=st.multiselect('Compare incident types',sorted(filtered.Incident_Type.unique()),default=sorted(filtered.Incident_Type.unique())[:2],key='compare_types')
        area_trend=filtered[filtered.Area.isin(area_choice)].groupby(['Month','Area']).size().reset_index(name='Incidents')
        st.plotly_chart(style_fig(px.line(area_trend,x='Month',y='Incidents',color='Area',markers=True,title='Area vs area: monthly reports')),use_container_width=True)
        type_trend=filtered[filtered.Incident_Type.isin(type_choice)].groupby(['Month','Incident_Type']).size().reset_index(name='Incidents')
        st.plotly_chart(style_fig(px.line(type_trend,x='Month',y='Incidents',color='Incident_Type',markers=True,title='Incident type vs incident type')),use_container_width=True)
        sev_trend=filtered.groupby(['Month','Severity']).size().reset_index(name='Incidents')
        st.plotly_chart(style_fig(px.area(sev_trend,x='Month',y='Incidents',color='Severity',title='Severity trends over time')),use_container_width=True)
        st.subheader('Correlation heatmap')
        numeric=filtered[['Resolution_Hours','Citizen_Rating','Estimated_Cost_INR']].corr(numeric_only=True)
        st.plotly_chart(style_fig(px.imshow(numeric,text_auto='.2f',zmin=-1,zmax=1,color_continuous_scale='RdBu_r',title='Numeric variable correlation')),use_container_width=True)
        st.subheader('Area drill-down')
        chosen=st.selectbox('Choose an area',sorted(filtered.Area.unique()),key='drill_area'); sub=filtered[filtered.Area.eq(chosen)]
        a,b,c,d=st.columns(4); a.metric('Incidents',f'{len(sub):,}'); b.metric('Estimated cost',f'₹{sub.Estimated_Cost_INR.sum():,.0f}'); b2=sub[sub.Status.eq('Resolved')].Resolution_Hours.median(); c.metric('Median resolution',f'{b2:.1f} hrs' if pd.notna(b2) else 'N/A'); r=sub[sub.Status.eq('Resolved')].Citizen_Rating.mean(); d.metric('Citizen rating',f'{r:.2f}/5' if pd.notna(r) else 'N/A')
        st.write('Top incident categories'); st.dataframe(sub.Incident_Type.value_counts().rename_axis('Incident type').reset_index(name='Count'),use_container_width=True,hide_index=True)
        st.write('Cost breakdown'); st.plotly_chart(style_fig(px.pie(sub,names='Incident_Type',values='Estimated_Cost_INR',hole=.45,title='Estimated cost by incident type')),use_container_width=True)
    # ---------- MAP ----------
    elif page=='🗺️ Incident Map':
        st.subheader('Geographic incident hotspots')
        m1,m2=st.columns(2)
        with m1: map_sev=st.multiselect('Map severity', ['Low','Medium','High','Critical'],default=['Low','Medium','High','Critical'],key='map_severity')
        with m2: map_types=st.multiselect('Map incident types',sorted(df.Incident_Type.unique()),default=sorted(df.Incident_Type.unique()),key='map_types')
        mapped=filtered[filtered.Severity.isin(map_sev)&filtered.Incident_Type.isin(map_types)]
        if mapped.empty: st.warning('No map points match those selections.')
        else:
            fig=px.scatter_mapbox(mapped,lat='Latitude',lon='Longitude',color='Severity',size='Estimated_Cost_INR',size_max=18,hover_name='Incident_ID',hover_data=['Area','Incident_Type','Status','Estimated_Cost_INR','Reported_At'],zoom=10,mapbox_style='open-street-map',title='Incident locations (illustrative coordinates)',color_discrete_map={'Low':'#38BDF8','Medium':'#FBBF24','High':'#FB923C','Critical':'#F43F5E'})
            fig.update_layout(template=template,paper_bgcolor='rgba(0,0,0,0)',font_color=fg,margin=dict(l=0,r=0,t=50,b=0),height=650)
            st.plotly_chart(fig,use_container_width=True)
            hot=mapped.Area.value_counts().rename_axis('Area').reset_index(name='Incidents'); st.subheader('Hotspot ranking'); st.dataframe(hot,use_container_width=True,hide_index=True)
            st.caption('Coordinates are generated around illustrative area centers for map demonstration; they are not real incident addresses.')
    # ---------- PREDICTIONS ----------
    elif page=='🤖 Predictions':
        st.subheader('Future Predictions')
        st.write('Random Forest regression forecasts daily report volume for the next 30 days from historical patterns. Forecasts are illustrative and should not be treated as operational predictions.')
        daily=df.set_index('Reported_At').resample('D').size().rename('Incidents').reset_index().rename(columns={'Reported_At':'Date'})
        daily['dayofweek']=daily.Date.dt.dayofweek; daily['dayofyear']=daily.Date.dt.dayofyear; daily['month']=daily.Date.dt.month; daily['lag_1']=daily.Incidents.shift(1); daily['lag_7']=daily.Incidents.shift(7); daily['rolling_7']=daily.Incidents.shift(1).rolling(7).mean(); model_data=daily.dropna().copy()
        features=['dayofweek','dayofyear','month','lag_1','lag_7','rolling_7']; split=max(30,int(len(model_data)*.8)); train=model_data.iloc[:split]; test=model_data.iloc[split:]
        model=RandomForestRegressor(n_estimators=250,max_depth=8,min_samples_leaf=2,random_state=42,n_jobs=-1); model.fit(train[features],train.Incidents)
        mae=mean_absolute_error(test.Incidents,model.predict(test[features])) if len(test) else np.nan
        st.metric('Back-test MAE (daily incidents)',f'{mae:.2f}' if pd.notna(mae) else 'N/A',help='Mean absolute error on a time-ordered holdout set; lower is better.')
        history=daily[['Date','Incidents']].copy(); last_date=history.Date.max(); recent=list(history.Incidents.tail(7).astype(float)); forecasts=[]
        for i in range(1,31):
            dt=last_date+pd.Timedelta(days=i); row=pd.DataFrame([{'dayofweek':dt.dayofweek,'dayofyear':dt.dayofyear,'month':dt.month,'lag_1':recent[-1],'lag_7':recent[-7] if len(recent)>=7 else recent[0],'rolling_7':float(np.mean(recent[-7:]))}]); pred=max(0,int(round(model.predict(row[features])[0]))); forecasts.append({'Date':dt,'Predicted incidents':pred}); recent.append(float(pred))
        forecast=pd.DataFrame(forecasts); hist_plot=history.tail(90).copy(); hist_plot['Series']='Historical'; forecast_plot=forecast.rename(columns={'Predicted incidents':'Incidents'}); forecast_plot['Series']='Forecast'; chart=pd.concat([hist_plot,forecast_plot],ignore_index=True)
        st.plotly_chart(style_fig(px.line(chart,x='Date',y='Incidents',color='Series',markers=False,title='Historical daily incidents and next 30-day forecast',color_discrete_map={'Historical':accent,'Forecast':'#F59E0B'})),use_container_width=True)
        c1,c2,c3=st.columns(3); c1.metric('Next 30-day total',f"{int(forecast['Predicted incidents'].sum()):,}"); c2.metric('Average per day',f"{forecast['Predicted incidents'].mean():.1f}"); c3.metric('Peak forecast day',str(forecast.loc[forecast['Predicted incidents'].idxmax(),'Date'].date()))
        st.dataframe(forecast,use_container_width=True,hide_index=True); st.download_button('Download forecast CSV',forecast.to_csv(index=False).encode(),file_name='civicpulse_30_day_forecast.csv',mime='text/csv')
        st.subheader('Machine learning risk score')
        ml=df.dropna(subset=['Resolution_Hours']).copy(); ml['Delayed']= (ml.Resolution_Hours>24).astype(int); ml['Hour']=ml.Reported_At.dt.hour
        cat=['Area','Incident_Type','Severity','Report_Source']; X=pd.get_dummies(ml[cat+['Estimated_Cost_INR','Hour']],columns=cat); y=ml.Delayed
        if y.nunique()>1:
            clf=RandomForestClassifier(n_estimators=150,max_depth=8,class_weight='balanced',random_state=42); clf.fit(X,y)
            selected_area=st.selectbox('Risk score area',sorted(df.Area.unique()),key='risk_area'); selected_type=st.selectbox('Risk score incident type',sorted(df.Incident_Type.unique()),key='risk_type'); selected_sev=st.selectbox('Risk score severity',['Low','Medium','High','Critical'],index=2,key='risk_sev'); cost=float(st.number_input('Estimated cost for risk example (₹)',min_value=0.0,value=15000.0,step=1000.0)); hour=int(st.slider('Report hour',0,23,12,key='risk_hour'))
            example=pd.DataFrame([{'Area':selected_area,'Incident_Type':selected_type,'Severity':selected_sev,'Report_Source':'Mobile App','Estimated_Cost_INR':cost,'Hour':hour}]); example=pd.get_dummies(example,columns=cat); example=example.reindex(columns=X.columns,fill_value=0); prob=float(clf.predict_proba(example)[0][1]); st.metric('Estimated delayed-resolution risk',f'{prob:.1%}',help='Model estimates the probability that resolution time exceeds 24 hours, based on synthetic historical data.')
            st.caption('This score is a demonstration of a machine-learning workflow, not a validated service-level prediction.')
    # ---------- PERFORMANCE ----------
    elif page=='📈 Performance':
        st.subheader('Incident resolution performance')
        resolved=filtered[filtered.Status.eq('Resolved')]
        median=resolved.Resolution_Hours.median(); sla=resolved.Resolution_Hours.le(24).mean()
        area_stats=resolved.groupby('Area').agg(Resolved=('Incident_ID','count'),Median_Hours=('Resolution_Hours','median'),SLA_Achievement=('SLA_Met','mean'),Avg_Rating=('Citizen_Rating','mean')).reset_index()
        a,b,c,d=st.columns(4); a.metric('Median resolution',f'{median:.1f} hrs' if pd.notna(median) else 'N/A'); b.metric('SLA achievement ≤24h',f'{sla:.1%}' if pd.notna(sla) else 'N/A'); best=area_stats.sort_values('Median_Hours').iloc[0].Area if len(area_stats) else 'N/A'; c.metric('Fastest area',best); slow=resolved.groupby('Incident_Type').Resolution_Hours.median().sort_values(ascending=False); d.metric('Slowest category',slow.index[0] if len(slow) else 'N/A')
        st.plotly_chart(style_fig(px.bar(area_stats.sort_values('Median_Hours'),x='Median_Hours',y='Area',orientation='h',color='SLA_Achievement',title='Median resolution time by area',text_auto='.1f')),use_container_width=True)
        st.subheader('Area performance table'); st.dataframe(area_stats.sort_values('Median_Hours'),use_container_width=True,hide_index=True)
        st.subheader('Resolution time by incident category'); cat_perf=resolved.groupby('Incident_Type',as_index=False).agg(Median_Hours=('Resolution_Hours','median'),Mean_Hours=('Resolution_Hours','mean'),Resolved=('Incident_ID','count')).sort_values('Median_Hours',ascending=False); st.dataframe(cat_perf,use_container_width=True,hide_index=True)
        st.subheader('Anomaly detection — unusual incident spikes')
        daily=filtered.set_index('Reported_At').resample('D').size().rename('Incidents').reset_index();
        if len(daily)>=20:
            iso=IsolationForest(contamination=.05,random_state=42); daily['Anomaly']=iso.fit_predict(daily[['Incidents']]); daily['Flag']=np.where(daily.Anomaly==-1,'Anomaly','Typical'); fig=px.scatter(daily,x='Reported_At',y='Incidents',color='Flag',color_discrete_map={'Anomaly':'#F43F5E','Typical':accent},title='Daily volume anomaly detection'); st.plotly_chart(style_fig(fig),use_container_width=True); st.dataframe(daily[daily.Anomaly.eq(-1)].sort_values('Incidents',ascending=False),use_container_width=True,hide_index=True)
        else: st.info('Select a date range with at least 20 days to run anomaly detection.')
    # ---------- REPORTS ----------
    elif page=='📄 Reports':
        st.subheader('Downloadable reports')
        st.markdown('Generate a report from the currently filtered dataset. PDF and Excel exports include summary information and grouped breakdowns.')
        a,b=st.columns(2)
        with a:
            st.markdown('#### 📄 PDF summary'); st.write('Executive summary, reporting period, KPIs, and area breakdown.')
            try: pdf_bytes=make_pdf(filtered); st.download_button('Download PDF summary',pdf_bytes,file_name='civicpulse_incident_summary.pdf',mime='application/pdf',use_container_width=True)
            except Exception as e: st.error(f'PDF export unavailable: {e}. Check that reportlab is installed.')
        with b:
            st.markdown('#### 📊 Excel workbook'); st.write('Incident-level data plus area and incident-type summary sheets.')
            try: xlsx_bytes=make_excel(filtered); st.download_button('Download Excel report',xlsx_bytes,file_name='civicpulse_incident_report.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',use_container_width=True)
            except Exception as e: st.error(f'Excel export unavailable: {e}. Check that openpyxl is installed.')
        st.markdown('#### 📈 Export chart data / charts')
        monthly=filtered.groupby('Month').size().rename('Incidents').reset_index()
        st.download_button('Download monthly trend CSV',monthly.to_csv(index=False).encode(),file_name='civicpulse_monthly_trend.csv',mime='text/csv')
        fig=style_fig(px.line(monthly,x='Month',y='Incidents',markers=True,title='Monthly incident trend for export'))
        st.plotly_chart(fig,use_container_width=True)
        st.download_button('Download chart as interactive HTML',fig.to_html(include_plotlyjs='cdn').encode(),file_name='civicpulse_monthly_chart.html',mime='text/html')
        st.caption('Interactive HTML preserves the Plotly chart. Static PNG export requires an additional Kaleido/browser setup.')
        st.subheader('Executive summary'); st.markdown(insight_text(filtered))
        st.subheader('Preview of report data'); st.dataframe(filtered.head(100),use_container_width=True,hide_index=True)

st.sidebar.caption(f"Showing {len(filtered):,} of {len(df):,} source records")
st.caption('CivicPulse Pro • Synthetic demonstration dataset • Random Forest / Isolation Forest outputs are illustrative, not validated operational advice.')
