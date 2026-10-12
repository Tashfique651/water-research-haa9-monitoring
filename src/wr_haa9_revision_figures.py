"""Publication-grade vector figures for the frozen September 2026 revision.

Every panel returns its exact source table through the FigureBundle contract.
No illustrations are rasterized. Numerical plotting is independent of prose.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Ellipse, Polygon
import numpy as np
import pandas as pd
import wr_make_figures as base

R = "wr_results/haa9_revision/"
H5,H9,ACCENT,GRAY = base.ORANGE,base.TEAL,base.MAGENTA,base.MID


def read(name):
    return base.load_csv(R+name+".csv")


def panels(height=165):
    fig,axs=plt.subplots(2,2,figsize=(base.FULL_W,height*base.MM),layout="constrained")
    fig.set_constrained_layout_pads(w_pad=.10,h_pad=.10,wspace=.16,hspace=.16)
    return fig,axs.ravel()


def polish(bundle,minsize=8.5):
    for t in bundle.fig.findobj(mpl.text.Text):
        if t.get_fontsize()<minsize:t.set_fontsize(minsize)
    return bundle


def bundle(name,fig,caption,alt,sources):
    return polish(base.FigureBundle(name,fig,caption,alt,sources))


def title(ax,letter,label):
    base.panel_title(ax,letter,label)


def glyph(ax,kind,center,width=.18,height=.18,color=base.NAVY):
    """Small semantic, code-native vector artwork in axes coordinates."""
    cx,cy=center
    def line(points,**kw):
        xx,yy=zip(*[(cx+width*x,cy+height*y) for x,y in points])
        ax.plot(xx,yy,transform=ax.transAxes,color=kw.pop("color",color),lw=kw.pop("lw",1.2),**kw)
    def box(x,y,w,h,fc="white",ec=None,lw=1.1):
        ax.add_patch(Rectangle((cx+width*x,cy+height*y),width*w,height*h,transform=ax.transAxes,fc=fc,ec=ec or color,lw=lw))
    if kind in {"sample","comparator"}:
        offsets=[0] if kind=="sample" else [-.30,.30]
        for offset in offsets:
            col=color if kind=="sample" else (H5 if offset<0 else H9)
            box(offset-.17,-.40,.34,.72,base.PALE_TEAL,col)
            box(offset-.15,-.38,.30,.35,base.PALE_ORANGE if col==H5 else base.PALE_TEAL,col,lw=.5)
            box(offset-.20,.32,.40,.10,col,col)
            line([(offset-.10,.17),(offset+.10,.17)],color=col,lw=1.0)
    elif kind=="calendar":
        box(-.48,-.40,.96,.80,base.WHITE)
        box(-.48,.18,.96,.22,base.PALE_TEAL)
        for x in [-.27,.27]:line([(x,.28),(x,.52)],lw=1.6)
        for x in [-.28,0,.28]:
            for y in [-.06,-.25]:box(x-.055,y-.04,.11,.09,color,color,lw=0)
    elif kind=="clock":
        ax.add_patch(Ellipse(center,width*.91,height*.91,transform=ax.transAxes,fc=base.PALE_ORANGE,ec=color,lw=1.2))
        line([(0,.29),(0,0),(.23,-.11)],lw=1.5)
        for x0,y0,x1,y1 in [(0,.38,0,.44),(.38,0,.44,0),(0,-.38,0,-.44),(-.38,0,-.44,0)]:line([(x0,y0),(x1,y1)],lw=.8)
    elif kind=="endpoint":
        for scale,fc in [(1,base.PALE_TEAL),(.58,base.WHITE),(.17,color)]:
            ax.add_patch(Ellipse(center,width*scale,height*scale,transform=ax.transAxes,fc=fc,ec=color,lw=1.1))
        line([(-.64,0),(-.38,0)]);line([(.38,0),(.64,0)]);line([(0,-.64),(0,-.38)]);line([(0,.38),(0,.64)])
    elif kind=="resources":
        glyph(ax,"calendar",(cx+width*.18,cy+height*.02),width*.73,height*.80,color)
        glyph(ax,"sample",(cx-width*.41,cy-height*.10),width*.60,height*.88,H9)
    elif kind=="validation":
        box(-.59,-.30,.42,.63,base.PALE_ORANGE,H5)
        box(.16,-.30,.42,.63,base.PALE_TEAL,H9)
        line([(-.13,0),(.12,0)],lw=1.4)
        line([(.04,.07),(.12,0),(.04,-.07)],lw=1.4)
        ax.text(cx+width*.37,cy+height*.02,"?",transform=ax.transAxes,ha="center",va="center",fontsize=12,fontweight="bold",color=H9)
        for y in [.16,.03,-.10]:line([(-.51,y),(-.27,y)],color=H5,lw=.9)
    elif kind=="water":
        # A water sample, not a geographical logo or a molecular structure.
        points=[(0,.55),(-.30,.08),(-.35,-.12),(-.23,-.36),(0,-.45),(.23,-.36),(.35,-.12),(.30,.08)]
        ax.add_patch(Polygon([(cx+width*x,cy+height*y) for x,y in points],transform=ax.transAxes,fc=base.PALE_TEAL,ec=color,lw=1.2))
        line([(-.14,-.03),(-.18,-.17),(-.09,-.26)],color=base.WHITE,lw=1.8)


def legend(ax,**kw):
    kw.setdefault("fontsize",8.5)
    ax.legend(**kw)


def fig1():
    b=base.wr2_fig1()
    archives=b.fig.axes[0];archives.clear();archives.axis("off");archives.set(xlim=(0,1),ylim=(0,1));title(archives,"a","")
    archive_rows=[("UCMR4","Paired replay","sample",base.NAVY), ("SYR4","Sampling\ncontext","calendar",H9), ("ICR","Historical\ncheck","clock",H5), ("California","Routine\ncontext","water",ACCENT)]
    for x,(name,role,kind,color) in zip([.115,.375,.635,.895],archive_rows):
        glyph(archives,kind,(x,.65),width=.16,height=.24 if kind=="clock" else .42,color=color)
        archives.text(x,.30,name,ha="center",va="center",fontweight="bold",fontsize=9)
        archives.text(x,.08,role,ha="center",va="center",fontsize=8.5,color=GRAY)
    ax=b.fig.axes[3];ax.clear()
    title(ax,"d","Paired finite-grid design")
    ax.set(xlim=(0,2),ylim=(2,0),xticks=[.5,1.5],yticks=[.5,1.5])
    ax.set_xticklabels(["HAA5","HAA9"]);ax.set_yticklabels(["One group","Full grid"])
    labels=[["p00","p10"],["p01","p11"]]
    for i in range(2):
        for j in range(2):
            col=H9 if j else H5
            ax.add_patch(Rectangle((j+.04,i+.04),.92,.92,fc="white",ec=ACCENT if (i,j)==(1,1) else base.LIGHT,lw=1.5 if (i,j)==(1,1) else .8))
            for rr in range(4):
                for cc in range(3):
                    active=i==1 or rr==1
                    ax.scatter(j+.25+cc*.25,i+.18+rr*.14,s=45,marker="s" if j else "o",fc=col if active else base.WHITE,ec=col if active else base.LIGHT,lw=.8,zorder=3)
            ax.text(j+.5,i+.84,labels[i][j],ha="center",va="center",fontsize=10,fontweight="bold",color=col)
    ax.set_xlabel("Chemical information");ax.set_ylabel("Observed support")
    for sp in ax.spines.values():sp.set_visible(False)
    source=pd.DataFrame([dict(row=i,column=j,label=labels[i][j],panel=["HAA5","HAA9"][j],support=["one_group","full_grid"][i]) for i in range(2) for j in range(2)])
    archive_source=pd.DataFrame(archive_rows,columns=["archive","evidence_role","vector_glyph","color"])
    b.sources=[s for s in b.sources if s[0] not in ["a","d"]]+[("a",archive_source,"archive-specific evidence roles; glyphs are illustrative"),("d",source,"declared paired factorial design; schematic dots, not numerical sample counts")]
    b.caption="Figure 1. Evidence roles and finite-grid replay. (a) Four displayed archive roles; SYR3 is reported separately. (b) Administrative-group semantics. (c) Example irregular site-by-round grid. (d) Matched panel-by-support design."
    b.alt_text="Letter-only panels use sample, calendar, clock and water glyphs for archive roles; grouping fractions; an observed irregular grid; and a paired support matrix whose filled versus open dots show selected versus omitted schematic observations."
    for text in b.fig.axes[0].texts:
        if "External\nreplication" in text.get_text():text.set_text("Sampling\ncontext")
    for text in b.fig.axes[1].texts:
        if text.get_text()=="93.7%":text.set_x(.917)
    for text in b.fig.axes[2].texts:
        if "star:" in text.get_text():text.set_position((.97,.97));text.set_ha("right")
    return polish(b)


def fig2():
    b=base.wr2_fig2();ax=b.fig.axes[0];ax.clear();ax.axis("off");title(ax,"a","Nine measured species")
    names=["MCAA","DCAA","TCAA","MBAA","DBAA","BCAA","BDCAA","CDBAA","TBAA"]
    positions=[(.17,.73),(.50,.73),(.83,.73),(.33,.55),(.67,.55),(.17,.29),(.50,.29),(.83,.29),(.50,.10)]
    ax.text(.5,.93,"HAA5",ha="center",color=H5,fontweight="bold",fontsize=10)
    ax.text(.5,.42,"Additional four",ha="center",color=H9,fontweight="bold",fontsize=10)
    for i,(name,(x,y)) in enumerate(zip(names,positions)):
        ax.add_patch(FancyBboxPatch((x-.135,y-.067),.27,.134,boxstyle="round,pad=.005",fc=base.PALE_ORANGE if i<5 else base.PALE_TEAL,ec=H5 if i<5 else H9,transform=ax.transAxes))
        ax.text(x,y,name,ha="center",va="center",fontsize=9,fontweight="bold")
    species=pd.DataFrame(dict(species=names,HAA5_member=[True]*5+[False]*4,HAA9_member=[True]*9))
    b.sources=[s for s in b.sources if s[0]!="a"]+[("a",species,"nested released panel definitions")]
    for t in b.fig.axes[3].texts:
        if "61,193" in t.get_text():t.set_text("61,193 nonzero-HAA9 events\nwithin eligible complete grids")
    # Keep annotations out of endpoint bars.
    for t in b.fig.axes[2].texts:
        if "full-record" in t.get_text():t.remove()
    b.caption="Figure 2. Added species create a one-directional detection gap. (a) Nested analyte scope. (b) Matched HAA5-HAA9 geometry. (c) Candidate-panel capture using each system's full-record individual-sample maximum. (d) Extra4 fraction quantiles."
    b.alt_text="The nine species are named, then compared through matched concentrations, full-record candidate-panel capture and additional-four composition quantiles among nonzero events."
    return polish(b)


def fig3():
    b=base.wr2_fig3();summary=read("oracle_interaction_summary")
    # Replace inherited primary intervals with canonical paired audit values.
    a=b.fig.axes[0];a.clear();title(a,"a","Reference-conditioned capture")
    f=summary[(summary.cohort=="reference_positive")&summary.estimand.isin(["y00","y10","y01","y11"])].set_index("estimand").loc[["y00","y10","y01","y11"]].reset_index()
    x=np.arange(4)
    base._errorbar_vertical(a,x,f.estimate,f.ci_lower,f.ci_upper,colors=[H5,H9,H5,H9],markers=["o","s","o","s"])
    a.set_xticks(x,["HAA5\none group","HAA9\none group","HAA5\nfull grid","HAA9\nfull grid"]);a.set_ylim(0,1.15);a.set_ylabel("Conditional capture");base.percent_axis(a);base.clean_axis(a)
    for j,v in enumerate(f.estimate):a.text(j,v+.04,f"{v:.1%}"+ ("*" if j==3 else ""),ha="center",fontsize=9,fontweight="bold")
    a.text(.03,.94,"* Defined by reference",transform=a.transAxes,ha="left",va="top",color=ACCENT,fontsize=8.5)
    scales=next(frame.copy() for selector,frame,path in b.sources if selector=="b")
    canonical=summary[(summary.cohort=="reference_positive")&(summary.estimand=="interaction")].iloc[0]
    for column in ["estimate","ci_lower","ci_upper"]:scales.loc[scales.measure.eq("Additive interaction"),column]=canonical[column]
    sb=b.fig.axes[1];sb.clear();title(sb,"b","Interaction across scales")
    yy=np.arange(len(scales));base.colored_forest(sb,scales.estimate,yy,scales.ci_lower,scales.ci_upper,[H9,base.NAVY,H5,base.PURPLE],size=46)
    for y,nul in zip(yy,scales["null"]):sb.scatter(nul,y,marker="|",s=180,color=GRAY,zorder=1)
    sb.set_yticks(yy,scales.measure);sb.invert_yaxis();sb.set_xlabel("Estimate; vertical tick is scale-specific null");base.clean_axis(sb,grid="x");sb.set_xlim(-.05,max(1.62,float(scales.ci_upper.max()+.08)))
    # Re-express conditional vs unconditional magnitude without mixed scales.
    d=b.fig.axes[3];d.clear();title(d,"d","Denominator changes magnitude")
    comp=summary[summary.estimand.eq("interaction")].copy().reset_index(drop=True)
    base.colored_forest(d,comp.estimate,[1,0],comp.ci_lower,comp.ci_upper,[ACCENT,base.NAVY],size=50)
    d.set_yticks([1,0],["769 reference-positive","4,904 eligible"]);d.set_ylim(-.6,1.6);d.set_xlim(0,.30);d.set_xlabel("Additive interaction");base.percent_axis(d,"x");base.clean_axis(d,grid="x")
    for y,row in zip([1,0],comp.itertuples()):d.text(row.ci_upper+.007,y,f"{100*row.estimate:.2f} pp",va="center",fontsize=9)
    b.sources=[s for s in b.sources if s[0] not in ["a","b","d"]]+[("a",f,R+"oracle_interaction_summary.csv"),("b",scales,"wr_results/playbook_extension/interaction_scale_metrics.csv; "+R+"oracle_interaction_summary.csv"),("d",comp,R+"oracle_interaction_summary.csv")]
    b.caption="Figure 3. Panel and observation changes jointly recover upper-tail detection. (a) Conditional capture among 769 reference-positive systems; p11 = 1 by definition. (b) Interaction scales. (c) Exhaustive repair classes. (d) Conditional and all-eligible interactions. Bars show 95% whole-PWS bootstrap intervals."
    b.alt_text="Four panels show a marked definitional capture cell, scale-dependent interaction, five repair classes and the much smaller all-eligible interaction."
    return polish(b)


def fig4():
    curve=read("budget_curve");support=read("support_budget_curve");cost=read("cost_resource_summary")
    fig,(a,b,c,d)=panels(168)
    title(a,"a","Matched physical-sample budgets")
    for p,col,marker in [("HAA5",H5,"o"),("HAA9",H9,"s")]:
        a.plot(curve.requested_samples,curve[p+"_capture"],color=col,marker=marker,markevery=2,label=p)
        a.fill_between(curve.requested_samples,curve[p+"_capture_ci_lower"],curve[p+"_capture_ci_upper"],color=col,alpha=.12)
    benchmark=curve.full_HAA5_capture.iloc[0];a.axhline(benchmark,ls="--",color=base.NAVY,label=f"Full HAA5 ({benchmark:.1%})")
    a.set(xlim=(1,16),ylim=(0,1.02),xticks=[1,4,8,12,16],xlabel="Requested physical samples",ylabel="Conditional capture");base.percent_axis(a);base.clean_axis(a);legend(a,loc="lower right")
    title(b,"b","Advantage over the full HAA5 record")
    b.fill_between(curve.requested_samples,curve.vs_full_HAA5_difference_simultaneous_lower*100,curve.vs_full_HAA5_difference_simultaneous_upper*100,color=H9,alpha=.18,label="95% simultaneous band")
    b.plot(curve.requested_samples,curve.vs_full_HAA5_difference*100,color=H9,marker="s",markevery=2)
    b.axhline(0,color=GRAY,lw=1)
    for k,y in [(4,-40),(8,40)]:
        row=curve[curve.requested_samples.eq(k)].iloc[0]
        b.scatter(k,row.vs_full_HAA5_difference*100,s=55,fc=ACCENT,ec="white",zorder=4)
        b.text(k,y,f"k={k}: {row.vs_full_HAA5_difference*100:+.2f} pp",ha="center",va="center",color=ACCENT,fontsize=9)
        b.plot([k,k],[y+(4 if y<0 else -4),row.vs_full_HAA5_difference*100],color=ACCENT,lw=.8,zorder=1)
    b.set(xlim=(1,16),ylim=(-47,48),xticks=[1,4,8,12,16],xlabel="Requested HAA9 samples",ylabel="Paired capture difference (pp)");base.clean_axis(b);legend(b,loc="lower right")
    title(c,"c","Temporal and spatial investments")
    for unit,label,ls in [("administrative_event_code_group","Groups","-"),("distribution_site","Sites","--")]:
        g=support[support.unit.eq(unit)].sort_values("requested_units")
        for p,col,mark in [("HAA5",H5,"o"),("HAA9",H9,"s")]:c.plot(g.requested_units,g[p+"_capture"],color=col,ls=ls,marker=mark,label=f"{p}: {label}")
    c.set(xlim=(.8,4.2),ylim=(0,1.03),xticks=[1,2,3,4],xlabel="Requested groups or sites",ylabel="Conditional capture");base.percent_axis(c);base.clean_axis(c);legend(c,loc="lower right",ncol=2,columnspacing=.6,handlelength=1.8)
    title(d,"d","Cost assumptions change the frontier")
    costplot=cost[cost.date_cost_assumption.eq(0)].copy()
    for mode,g in costplot.groupby("mode"):
        col=base.NAVY if mode=="hard" else H9
        d.plot(g.assay_ratio,g.difference*100,color=col,ls="--" if mode=="hard" else "-",label=mode.capitalize()+" budget")
        d.fill_between(g.assay_ratio,g.difference_ci_lower*100,g.difference_ci_upper*100,color=col,alpha=.10)
    d.axhline(0,color=GRAY,lw=1);d.set(xlabel="HAA9:HAA5 analytical cost ratio",ylabel="Gain over full HAA5 (pp)");base.clean_axis(d);legend(d,loc="lower left",ncol=2,columnspacing=.6,handlelength=1.8)
    caption="Figure 4. Broader chemistry recovers more signals with fewer samples. Among 769 reference-positive systems: (a) equal-sample capture; (b) HAA9 minus complete-grid HAA5; (c) round and fixed-site budgets; (d) assumed hard and expected assay-cost scenarios. The shaded band in (b) is simultaneous across the complete 32-contrast budget family (95% bootstrap coverage)."
    return bundle("Fig4",fig,caption,"Four panels show physical-sample capture, a multiplicity-aware paired budget advantage, separate temporal and spatial curves, and hypothetical cost frontiers.",[("a",curve,R+"budget_curve.csv"),("b",curve,R+"budget_curve.csv"),("c",support,R+"support_budget_curve.csv"),("d",costplot,R+"cost_resource_summary.csv")])


def fig5():
    corr=read("chemistry_source_correlations");mod=read("chemistry_environmental_models");cat=read("chemistry_category_summary");unc=read("chemistry_discordance_summary")
    fig,(a,b,c,d)=panels(177)
    title(a,"a","Concentration is not composition")
    cc=corr[(corr.cohort=="all_eligible")&(corr.nondetect_scenario=="midpoint")].copy()
    endpoints=["HAA5_full_max","HAA9_full_max","Extra4_max_ug_l","median_Extra4_fraction"]
    # Resolve the exact composition field from the source schema.
    comp=[v for v in cc.endpoint.unique() if "fraction" in v.lower()][0];endpoints[-1]=comp
    labels=["HAA5 maximum","HAA9 maximum","Extra4 maximum","Extra4 fraction"]
    cc=cc[cc.endpoint.isin(endpoints)]
    for j,(ind,col) in enumerate([("bromide",H9),("toc",H5)]):
        g=cc[cc.indicator.eq(ind)].set_index("endpoint").loc[endpoints]
        a.barh(np.arange(4)+(j-.5)*.32,g.spearman_rho,height=.30,color=col,label="Bromide" if j==0 else "TOC")
    a.axvline(0,color=GRAY,lw=1);a.set_yticks(range(4),labels);a.set_ylim(4.3,-.5);a.set(xlabel="Spearman correlation",xlim=(-.7,.9));base.clean_axis(a,grid="x");legend(a,loc="lower right",ncol=2)
    title(b,"b","Environmental association sensitivity")
    mm=mod[(mod.model=="conditional_invisibility")&mod.term.isin(["bromide","toc"])].copy()
    scenarios=["lower","midpoint","upper"]
    for j,sc in enumerate(scenarios):
        g=mm[mm.nondetect_scenario.eq(sc)].set_index("term").loc[["bromide","toc"]]
        y=np.arange(2)+(j-1)*.18
        base.colored_forest(b,g.odds_ratio,y,g.ci_lower,g.ci_upper,[base.CB[j]]*2,size=33)
    b.axvline(1,ymin=.14,color=GRAY,ls="--");b.set_xscale("log");b.set_yticks([0,1],["Bromide","TOC"]);b.invert_yaxis();b.set_xlabel("Adjusted odds ratio / transformed IQR");b.set_ylim(1.55,-.5);base.clean_axis(b,grid="x")
    b.set_xticks([.5,1,2,4],["0.5","1","2","4"]);b.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    b.legend(handles=[Line2D([0],[0],marker="o",color=base.CB[j],ls="",label=sc.capitalize()) for j,sc in enumerate(scenarios)],loc="lower right",fontsize=8.5,ncol=3,columnspacing=.6,handletextpad=.3)
    title(c,"c","Full-record concentration categories")
    categories=["neither","both","HAA9_only"]
    present=cat.category.unique().tolist();categories=[v for v in categories if v in present]+[v for v in present if v not in categories]
    variables=["HAA5_full_max","HAA9_full_max","Extra4_max_ug_l"]
    use=cat[cat.variable.isin(variables)]
    for j,(var,col,label) in enumerate(zip(variables,[H5,H9,ACCENT],["HAA5","HAA9","Extra4"])):
        g=use[use.variable.eq(var)].set_index("category").loc[categories];x=np.arange(len(categories))+(j-1)*.22
        c.errorbar(x,g["median"],yerr=[g["median"]-g.q25,g.q75-g["median"]],fmt="o",color=col,label=label,capsize=3)
    c.axhline(60,color=GRAY,ls="--");c.set_xticks(range(len(categories)),[v.replace("_"," ").replace("neither","Neither").replace("both","Both") for v in categories]);c.set_ylabel("System maximum (µg/L), median and IQR");c.set_ylim(0,125);base.clean_axis(c);legend(c,loc="upper left",ncol=3,columnspacing=.7,handletextpad=.3)
    title(d,"d","Assumed measurement envelopes")
    uu=unc[unc.cohort.eq("official_discordant")].copy();x=np.arange(len(uu))
    d.bar(x,uu.definite,color=H9,label="Definite discordance")
    d.bar(x,uu.possible_not_definite,bottom=uu.definite,color=base.PALE_ORANGE,edgecolor=H5,hatch="//",label="Uncertainty-sensitive")
    for xi,v in zip(x,uu.definite):d.text(xi,v/2,str(v),ha="center",va="center",color="white",fontweight="bold",fontsize=8.5)
    d.set_xticks(x,[f"{v:.0%}" for v in uu.quantified_relative_envelope]);d.set(xlabel="Assumed relative envelope on quantified values",ylabel="Official HAA9-only systems",ylim=(0,420));base.clean_axis(d);legend(d,loc="upper center",fontsize=8.5)
    caption="Figure 5. Chemical burden and discordance require separate interpretations. (a) Source-indicator correlations. (b) Censoring-sensitive associations with 95% HC0 CIs. (c) Absolute concentrations by detection category. (d) Definite discordance under assumed measurement envelopes used for sensitivity analysis."
    return bundle("Fig5",fig,caption,"Four panels separate environmental descriptors from shared chemical accounting, show nondetect-sensitive associations, category concentrations, and uncertainty-sensitive extra detections.",[("a",cc,R+"chemistry_source_correlations.csv"),("b",mm,R+"chemistry_environmental_models.csv"),("c",use,R+"chemistry_category_summary.csv"),("d",uu,R+"chemistry_discordance_summary.csv")])


def fig6():
    season=read("season_summary");ep=read("common_support_endpoint_summary")
    fig,(a,b,c,d)=panels(166)
    ss=season[(season.cohort=="reference_positive")&(season.requested_units==1)].copy()
    order=["administrative_event_code_group","collection_date","physical_sample"];ss=ss.set_index("unit").loc[order].reset_index()
    title(a,"a","The same seasonal selector for both panels")
    for j,(field,color,ls,label) in enumerate([("HAA5_uniform",H5,"--","HAA5 uniform"),("HAA5_warm",H5,"-","HAA5 warm"),("HAA9_uniform",H9,"--","HAA9 uniform"),("HAA9_warm",H9,"-","HAA9 warm")]):
        a.plot(range(3),ss[field],color=color,ls=ls,marker="o" if "HAA5" in field else "s",label=label)
    a.set_xticks(range(3),["Group","Date","Sample"]);a.set(ylabel="Conditional capture",ylim=(0,.72));base.percent_axis(a);base.clean_axis(a);legend(a,loc="upper right",ncol=2,columnspacing=.6)
    title(b,"b","Panel-by-season interaction")
    pp=season[(season.cohort=="reference_positive")&(season.unit=="physical_sample")].copy()
    base.colored_forest(b,pp.panel_by_season_interaction*100,range(4),pp.panel_by_season_interaction_ci_lower*100,pp.panel_by_season_interaction_ci_upper*100,[H9]*4,size=42)
    b.axvline(0,color=GRAY,ls="--");b.set_yticks(range(4),[f"{k} physical sample"+("s" if k>1 else "") for k in pp.requested_units]);b.invert_yaxis();b.set_xlabel("Extra panel advantage from season (pp)");base.clean_axis(b,grid="x")
    title(c,"c","Identical supported systems and windows")
    endpoints=["maximum","OEL_form","LRAA_form"]
    cc=ep[ep.reference.eq("all_common_support_PWS")].copy()
    for j,(p,color) in enumerate([("HAA5",H5),("HAA9",H9)]):
        g=cc[cc.panel.eq(p)].set_index("endpoint").loc[endpoints];x=np.arange(3)+(j-.5)*.23
        c.errorbar(x,g.capture_or_positive_fraction,yerr=[g.capture_or_positive_fraction-g.ci_lower,g.ci_upper-g.capture_or_positive_fraction],fmt="o" if p=="HAA5" else "s",color=color,label=p,capsize=3)
    c.set_xticks(range(3),["Maximum","OEL-form","LRAA-form"]);c.set(ylabel="Positive fraction of 2,487 PWSs",ylim=(0,.30));base.percent_axis(c);base.clean_axis(c);legend(c,loc="upper right")
    title(d,"d","Hold the reference group fixed")
    dd=ep[ep.reference.eq("fixed_maximum_HAA9_positive")].copy()
    for j,(p,color) in enumerate([("HAA5",H5),("HAA9",H9)]):
        g=dd[dd.panel.eq(p)].set_index("endpoint").loc[endpoints];x=np.arange(3)+(j-.5)*.23
        d.errorbar(x,g.capture_or_positive_fraction,yerr=[g.capture_or_positive_fraction-g.ci_lower,g.ci_upper-g.capture_or_positive_fraction],fmt="o" if p=="HAA5" else "s",color=color,label=p,capsize=3)
    d.set_xticks(range(3),["Maximum","OEL-form","LRAA-form"]);d.set(ylabel="Capture in 610 maximum-positive PWSs",ylim=(0,1.08));base.percent_axis(d);base.clean_axis(d);legend(d,loc="upper right")
    caption="Figure 6. Scheduling and aggregation modify the panel advantage. (a) Seasonal capture among 769 reference-positive systems. (b) Panel-by-season interactions with 95% whole-PWS bootstrap intervals. On identical four-quarter site windows: (c) positivity among 2,487 systems; (d) capture among 610 maximum-positive systems. Averaged endpoints are research analogues, not compliance determinations."
    return bundle("Fig6",fig,caption,"Four panels show shared seasonal selectors, their paired interaction, and endpoint comparisons with both observational support and reference denominator held fixed.",[("a",ss,R+"season_summary.csv"),("b",pp,R+"season_summary.csv"),("c",cc,R+"common_support_endpoint_summary.csv"),("d",dd,R+"common_support_endpoint_summary.csv")])


def fig7():
    # Promote the already frozen S3(a,b,d) evidence, not a new threshold analysis.
    full=base.load_csv("wr_results/core/threshold_factorial_with_intervals.csv")
    band=base.load_csv("wr_results/core/threshold_interaction_simultaneous_band.csv")
    spec=full[full.cohort_definition.eq("threshold_specific_reference_positive")].copy()
    fixed=full[full.cohort_definition.eq("fixed_B60_reference_positive_systems")].copy()
    rounds=base.load_csv("wr_results/core/round_definition_sensitivity.csv")
    rounds["label"]=rounds.analysis.map({"event_code":"Event code","event_code_year":"Event code + year","collection_date":"Collection date","facility_event_code_year":"Facility + event + year","physical_sample":"Physical sample"})
    fig,(a,b,c,d)=panels(152)
    title(a,"a","")
    for metric,color,label,marker in [("p00",base.NAVY,"HAA5 / one round","o"),("p10",H9,"HAA9 / one round","s"),("p01",H5,"HAA5 / complete grid","^"),("p11",ACCENT,"HAA9 / complete grid","D")]:
        a.fill_between(spec.threshold_ug_l,spec[metric+"_ci_lower"],spec[metric+"_ci_upper"],color=color,alpha=.10)
        a.plot(spec.threshold_ug_l,spec[metric],color=color,marker=marker,ms=3,markevery=2,label=label)
    a.set(xlabel="Analytical benchmark (µg/L)",ylabel="Conditional capture",xlim=(18,102),ylim=(0,1.05),xticks=[20,40,60,80,100])
    base.percent_axis(a);base.clean_axis(a);legend(a,loc="upper center",bbox_to_anchor=(.5,-.23),ncol=2,fontsize=8.5)
    title(b,"b","")
    b.fill_between(band.threshold_ug_l,100*band.simultaneous_ci_lower,100*band.simultaneous_ci_upper,color=base.SKY,alpha=.22,label="95% simultaneous band")
    b.fill_between(band.threshold_ug_l,100*band.pointwise_ci_lower,100*band.pointwise_ci_upper,color=base.NAVY,alpha=.20,label="95% pointwise interval")
    b.plot(band.threshold_ug_l,100*band.interaction,color=base.NAVY,marker="o",ms=3,markevery=2)
    b.axhline(0,color=GRAY,lw=.9);b.axvline(60,ymin=.28,ymax=.98,color=ACCENT,ls=":",lw=.9)
    b.set(xlabel="Analytical benchmark (µg/L)",ylabel="Additive interaction (pp)",xlim=(18,102),xticks=[20,40,60,80,100],ylim=(-15,40))
    base.clean_axis(b);legend(b,loc="lower right",fontsize=8.5)
    title(c,"c","")
    c.plot(spec.threshold_ug_l,spec.p00,color=base.NAVY,marker="o",ms=3,markevery=2,label="Threshold-specific cohort")
    c.plot(fixed.threshold_ug_l,fixed.p00,color=H5,marker="s",ms=3,markevery=2,label="Fixed B60 cohort")
    c.axvline(60,ymax=.74,color=ACCENT,ls=":",lw=.9)
    c.set(xlabel="Analytical benchmark (µg/L)",ylabel="One-round HAA5 capture",xlim=(18,102),ylim=(0,1),xticks=[20,40,60,80,100])
    base.percent_axis(c);base.clean_axis(c);legend(c,loc="upper right",fontsize=8.5)
    title(d,"d","")
    y=np.arange(len(rounds))[::-1]
    base.colored_forest(d,100*rounds.interaction,y,100*rounds.interaction_ci_lower,100*rounds.interaction_ci_upper,[base.NAVY,H9,H5,ACCENT,base.PURPLE],size=44)
    d.set_yticks(y,rounds.label);d.set(xlabel="B60 additive interaction (pp)",xlim=(7,36),ylim=(-.6,4.6));base.clean_axis(d,grid="x")
    caption="Figure 7. Complementarity depends on benchmark and estimand but persists across observation definitions. (a) Threshold-specific conditional capture for four factorial cells. (b) Additive interaction with pointwise 95% whole-PWS bootstrap intervals and a simultaneous 95% bootstrap band. (c) One-round HAA5 capture under threshold-specific and fixed-B60 estimands. (d) B60 interaction under alternative reduced-support definitions. B60 is an analytical benchmark, not an HAA9 regulatory or health threshold."
    return bundle("Fig7",fig,caption,"Four letter-only panels show four conditional capture cells across benchmarks, additive interaction with pointwise and simultaneous uncertainty, divergence of threshold-specific and fixed-B60 one-round HAA5 capture, and positive B60 interactions under five observation-unit definitions. Interaction changes sign at lower benchmarks; it is not benchmark-invariant.",[("a",spec,"wr_results/core/threshold_factorial_with_intervals.csv"),("b",band,"wr_results/core/threshold_interaction_simultaneous_band.csv"),("c",full,"wr_results/core/threshold_factorial_with_intervals.csv"),("d",rounds,"wr_results/core/round_definition_sensitivity.csv")])


def fig8():
    ops=read("forward_operating_points");contrasts=read("comparator_paired_contrasts");scores=read("forward_score_metrics");cal=read("forward_probability_calibration")
    fig,(a,b,c,d)=panels(172)
    title(a,"a","Forward detection versus follow-up")
    aa=ops[ops.strategy.str.startswith("scalar_")].copy()
    for p,col in [("HAA5",H5),("HAA9",H9)]:
        g=aa[aa.strategy.str.startswith("scalar_"+p+"_workload_")].sort_values("followup_fraction")
        a.plot(g.followup_fraction,g.sensitivity,color=col,marker="o" if p=="HAA5" else "s",label=p)
    a.set(xlabel="Actual held-out follow-up fraction",ylabel="Later-period sensitivity",xlim=(0,.34),ylim=(0,.75));base.percent_axis(a,"x");base.percent_axis(a);base.clean_axis(a);legend(a,loc="lower right")
    title(b,"b","All declared workload contrasts")
    bb=contrasts[(contrasts.strategy=="scalar_HAA9_minus_scalar_HAA5")&(contrasts.metric=="sensitivity")].sort_values("target_training_workload")
    x=bb.target_training_workload.to_numpy()
    b.fill_between(x,100*bb.simultaneous_ci_lower,100*bb.simultaneous_ci_upper,color=H9,alpha=.15,label="95% simultaneous band")
    b.errorbar(x,100*bb.estimate,yerr=[100*(bb.estimate-bb.ci_lower),100*(bb.ci_upper-bb.estimate)],fmt="s-",color=H9,capsize=3,label="Estimate / pointwise interval")
    b.axhline(0,color=GRAY,lw=1);b.set(xlabel="Training-selected workload target",ylabel="HAA9 minus HAA5 sensitivity (pp)");base.percent_axis(b,"x");base.clean_axis(b);legend(b,loc="lower right")
    title(c,"c","Species-resolved information")
    cc=scores[scores.strategy.isin(["logistic_five_species","logistic_nine_species"])&scores.metric.isin(["AUROC","average_precision"])].copy()
    for j,(s,col,label) in enumerate([("logistic_five_species",H5,"Five species"),("logistic_nine_species",H9,"Nine species")]):
        g=cc[cc.strategy.eq(s)].set_index("metric").loc[["AUROC","average_precision"]];x=np.arange(2)+(j-.5)*.15
        c.errorbar(x,g.estimate,yerr=[g.estimate-g.ci_lower,g.ci_upper-g.estimate],fmt="o" if j==0 else "s",color=col,label=label,capsize=3)
    c.set_xticks([0,1],["AUROC","Average precision"]);c.set(ylabel="Held-out score",ylim=(.25,.92));base.clean_axis(c);legend(c,loc="upper right")
    title(d,"d","Probability calibration")
    dd=cal[cal.strategy.isin(["logistic_five_species","logistic_nine_species"])].copy()
    for s,col,label in [("logistic_five_species",H5,"Five species"),("logistic_nine_species",H9,"Nine species")]:
        g=dd[dd.strategy.eq(s)].sort_values("mean_predicted_probability")
        brier=scores[(scores.strategy==s)&(scores.metric=="Brier_score")].estimate.iloc[0]
        d.plot(g.mean_predicted_probability,g.observed_positive_fraction,color=col,marker="o" if s=="logistic_five_species" else "s",label=f"{label}; Brier {brier:.3f}")
    d.plot([0,1],[0,1],ls="--",color=GRAY,lw=1);d.set(xlabel="Predicted probability",ylabel="Observed later-positive fraction",xlim=(0,1),ylim=(0,1));base.percent_axis(d,"x");base.percent_axis(d);base.clean_axis(d);legend(d,loc="upper left")
    caption="Figure 8. Later-period screening yields workload-dependent gains. Among 2,636 date-separated systems, including 420 later positives: (a) achieved workload and sensitivity; (b) paired scalar sensitivity gains; (c) five- versus nine-species discrimination; (d) probability calibration. Simultaneous 95% bootstrap bands in (b) cover five declared workload targets, conditional on fitted rules."
    return bundle("Fig8",fig,caption,"Four panels challenge HAA9 using calibrated HAA5 thresholds and species-resolved models under strictly chronological grouped internal validation, retaining every declared workload target.",[("a",aa,R+"forward_operating_points.csv"),("b",bb,R+"comparator_paired_contrasts.csv"),("c",cc,R+"forward_score_metrics.csv"),("d",dd,R+"forward_probability_calibration.csv"),("d",scores[(scores.metric=="Brier_score")&scores.strategy.isin(["logistic_five_species","logistic_nine_species"])],R+"forward_score_metrics.csv")])


def fig9():
    curve=read("budget_curve");fw=read("comparator_paired_contrasts");row=curve[curve.requested_samples.eq(8)].iloc[0]
    fig=plt.figure(figsize=(base.FULL_W,121*base.MM),layout="constrained")
    gs=fig.add_gridspec(1,2,width_ratios=[.9,1.1],wspace=.14);a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1])
    title(a,"a","What the archive supports")
    values=[row.full_HAA5_capture,row.HAA9_capture];a.barh([1,0],values,color=[H5,H9],height=.55)
    a.set_yticks([1,0],["Full HAA5\nrecord","Up to eight\nHAA9 samples"]);a.set(xlim=(0,1),ylim=(-.8,1.8),xlabel="Conditional finite-grid capture");base.percent_axis(a,"x");base.clean_axis(a,grid="x")
    for y,v in zip([1,0],values):a.text(v-.025,y,f"{v:.2%}",ha="right",va="center",fontweight="bold",color="white",fontsize=12)
    a.text(.5,.90,"+22.07 percentage points",transform=a.transAxes,ha="center",color=ACCENT,fontweight="bold",fontsize=10)
    title(b,"b","Translate evidence into a monitoring decision")
    b.axis("off");b.set(xlim=(0,1),ylim=(0,1))
    # Four constraints meet at the decision. This is not a staged flowchart.
    for xs,ys in [([.40,.20,.20],[.54,.54,.69]),([.60,.80,.80],[.54,.54,.69]),([.40,.20,.20],[.46,.46,.34]),([.60,.80,.80],[.46,.46,.34])]:
        b.plot(xs,ys,color=base.LIGHT,lw=2,transform=b.transAxes,zorder=1)
    b.add_patch(Ellipse((.5,.50),.31,.19,transform=b.transAxes,fc=base.PALE_TEAL,ec=H9,lw=1.2,zorder=2))
    b.text(.5,.50,"Monitoring\nchoice",ha="center",va="center",fontsize=11,fontweight="bold",zorder=3)
    glyph(b,"endpoint",(.20,.80),width=.19,height=.14,color=base.NAVY)
    glyph(b,"comparator",(.80,.80),width=.25,height=.20,color=H9)
    glyph(b,"resources",(.20,.23),width=.26,height=.19,color=base.NAVY)
    glyph(b,"validation",(.80,.23),width=.24,height=.19,color=H9)
    for x,y,label in [(.20,.95,"Endpoint"),(.80,.95,"Calibrated\ncomparators"),(.20,.065,"Actual resources"),(.80,.065,"Independent\nvalidation needed")]:
        b.text(x,y,label,ha="center",va="center",fontsize=10,fontweight="bold",color=base.NAVY)
    statements=[("Endpoint","endpoint","Observed HAA9 signal, not health risk"),("Calibrated comparators","comparator","Compare HAA5 and broader chemistry"),("Actual resources","resources","Sample, date, assay and follow-up costs"),("Independent validation needed","validation","Strict chronology, then independent data; question mark denotes unresolved independent validation")]
    src=pd.DataFrame(statements,columns=["decision_constraint","vector_glyph","boundary"])
    caption="Figure 9. Conditions for useful monitoring expansion. (a) Eight-sample detection gain. (b) Objective, comparator, cost and validation requirements for implementation."
    return bundle("Fig9",fig,caption,"Letter-only panels show the eight-sample capture advantage and four icon-led constraints around a monitoring choice: endpoint, calibrated comparators, actual resources and independent validation. Orthogonal connectors convey joint requirements, not sequential proof.",[("a",curve[curve.requested_samples.eq(8)],R+"budget_curve.csv"),("b",src,"evidence-constrained synthesis of the completed analyses")])


def graphical_abstract():
    curve=read("budget_curve");row=curve[curve.requested_samples.eq(8)].iloc[0]
    fig,axs=plt.subplots(1,3,figsize=(base.GA_W,76*base.MM),gridspec_kw={"width_ratios":[1,1.05,1.05]})
    fig.subplots_adjust(left=.035,right=.985,bottom=.22,top=.78,wspace=.36)
    fig.suptitle("Broader panels at finite sample budgets",fontsize=15,fontweight="bold",y=.96)
    a,b,c=axs
    for ax in axs:title(ax,"abc"[list(axs).index(ax)],"")
    for ax in [a,b]:ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
    # Paired assay glyphs show nested analyte panels, not different water samples.
    for center,n,label in [(.20,5,"HAA5"),(.79,9,"HAA9")]:
        a.add_patch(FancyBboxPatch((center-.18,.30),.36,.50,boxstyle="round,pad=.008,rounding_size=.05",fc=base.PALE_TEAL,ec=base.NAVY,lw=1.2))
        a.add_patch(Rectangle((center-.10,.80),.20,.08,fc=base.WHITE,ec=base.NAVY,lw=1.1))
        a.add_patch(Rectangle((center-.13,.88),.26,.055,fc=base.NAVY,ec=base.NAVY,lw=1))
        for i in range(n):
            x=center-.105+(i%3)*.105;y=.69-(i//3)*.125
            a.scatter(x,y,s=36,color=H5 if i<5 else H9,marker="o" if i<5 else "s",edgecolor="white",lw=.5)
        a.text(center,.17,label,ha="center",fontsize=11,fontweight="bold",color=H5 if n==5 else H9)
    base.straight_arrow(a,(.42,.55),(.57,.55),color=base.NAVY,lw=1.3)
    a.text(.5,.04,"+4 measured acids",ha="center",fontsize=11)
    for y,v,label,col in [(.67,row.HAA9_capture,"Up to 8 HAA9 samples",H9),(.36,row.full_HAA5_capture,"Complete HAA5 grid",H5)]:
        b.add_patch(Rectangle((.03,y-.05),v*.93,.14,fc=col));b.text(.04,y+.16,label,ha="left",fontsize=11);b.text(.03+v*.465,y+.02,f"{v:.1%}",ha="center",va="center",fontsize=13,fontweight="bold",color="white")
    b.text(.5,.09,"+22.07 pp",ha="center",fontsize=13,color=ACCENT,fontweight="bold")
    b.text(.5,-.14,"769 reference-positive PWSs\nB = 60 µg/L",ha="center",va="center",fontsize=11,color=GRAY)
    definitions=pd.DataFrame(dict(panel=["HAA5","HAA9"],species=[5,9]))
    forward=read("comparator_paired_contrasts");forward=forward[(forward.strategy=="scalar_HAA9_minus_scalar_HAA5")&(forward.metric=="sensitivity")]
    x=100*forward.target_training_workload
    c.fill_between(x,100*forward.simultaneous_ci_lower,100*forward.simultaneous_ci_upper,color=H9,alpha=.18)
    c.plot(x,100*forward.estimate,"s-",color=H9,ms=4,lw=1.3)
    c.axhline(0,color=GRAY,lw=.9)
    c.set(xlim=(3,32),ylim=(-3,10),xticks=[5,15,30],yticks=[0,5,10],xlabel="Training target (%)",ylabel="Sensitivity gain (pp)")
    base.clean_axis(c);c.tick_params(labelsize=11);c.xaxis.label.set_size(11);c.yaxis.label.set_size(11)
    fig.text(.81,.02,"Internal: HAA9 − calibrated HAA5",ha="center",fontsize=11,color=GRAY)
    caption="Graphical abstract. (a) Paired assay glyphs show five core and four additional measured acids. (b) At the B60 analytical benchmark, up to eight HAA9 samples capture 79.80% of 769 reference-positive systems versus 57.74% for the complete observed HAA5 grid; the 22.07-point difference has a simultaneous 95% bootstrap band of 17.85–26.28 points across the 32-contrast budget family. (c) All five training-selected workload targets show the paired HAA9-minus-HAA5 forward sensitivity contrast with a simultaneous 95% bootstrap band; gains are modest and workload-dependent. Grouped chronological evaluation is internal and retrospective, not independent confirmation."
    return bundle("Graphical_Abstract",fig,caption,"Letter-only panels show paired five- and nine-analyte assay glyphs joined by a straight arrow, the eight-sample capture advantage, and all five forward sensitivity contrasts with a zero line and simultaneous uncertainty band.",[("a",definitions,"released panel definitions; paired assay glyphs are schematic"),("b",curve[curve.requested_samples.eq(8)],R+"budget_curve.csv"),("c",forward,R+"comparator_paired_contrasts.csv")])


def supplementary9():
    b=base.wr2_figs9()
    b.fig.set_size_inches(base.FULL_W,195*base.MM)
    policy_ax=b.fig.axes[2];policy_ax.set_ylim(-2.1,7.5)
    panel_ax=b.fig.axes[3];panel_ax.set_ylim(0,56)
    for line in panel_ax.lines:
        if len(line.get_xdata())==2 and line.get_xdata()[0]==line.get_xdata()[1]:line.set_ydata([0,.73])
    b.caption=b.caption.replace("prespecified","retrospective").replace("prospective random, first-group, and warm-season rules","implementable random, first-group and warm-season rules evaluated retrospectively")
    b.caption+=" The shared highest-HAA9 selector was independently recomputed with zero archived-row discrepancies; its equality with full HAA5 is empirical, not guaranteed. Larger panel-enumeration interaction is not better monitoring performance."
    for text in b.fig.findobj(mpl.text.Text):
        if "Prospective rule" in text.get_text():text.set_text(text.get_text().replace("Prospective rule","Implementable rule"))
    return b


def supplementary6():
    b=base.figs6();ax=b.fig.axes[3];ax.clear()
    source=read("support_direction_weighting");source=source[source.estimand.eq("interaction")].copy()
    title(ax,"d","Pooled-direction weighting")
    base.colored_forest(ax,source.estimate,[1,0],source.ci_lower,source.ci_upper,[base.NAVY,H9],size=44)
    ax.set_yticks([1,0],["Equal PWS","Equal direction row"]);ax.set_ylim(-.6,1.6);ax.set_xlim(0,.08)
    ax.set_xlabel("B60 interaction; PWS-cluster interval");base.percent_axis(ax,"x");base.clean_axis(ax,grid="x")
    b.sources=[s for s in b.sources if s[0]!="d"]+[("d",source,R+"support_direction_weighting.csv")]
    b.caption="Figure S6. Disjoint-round evaluation and weighting. (a) Late groups define reference positivity and early groups evaluate capture. (b) Early groups define reference positivity and late groups evaluate capture. (c) Chronological interactions vary with threshold. (d) Alternating-fold pooling includes 1,021 direction rows from 765 PWSs; averaging within PWS before pooling gives a different estimand from equal direction-row weighting. Both intervals resample PWS clusters."
    b.alt_text="Chronological cell and interaction curves are accompanied by an explicit comparison between equal-PWS and equal-direction weighting of pooled alternating folds."
    return b


def builders():
    return [graphical_abstract,fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8,fig9,
            base.figs1,base.figs2,base.figs3,base.figs4,base.figs5,supplementary6,base.figs7,base.figs8,supplementary9,base.wr2_figs10_final]


def audit_main_text_layout():
    """Report text/text intersections, distinct from intended data mark crossings."""
    base.configure_style();records=[]
    for builder in builders():
        item=builder();base._validate_figure_design(item);item.fig.canvas.draw();renderer=item.fig.canvas.get_renderer()
        inactive_ticks=set()
        for ax in item.fig.axes:
            for axis,limits in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
                lo,hi=sorted(limits)
                for tick in axis.get_major_ticks():
                    if not lo-1e-12 <= tick.get_loc() <= hi+1e-12:
                        inactive_ticks.update([id(tick.label1),id(tick.label2)])
        text=[t for t in item.fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text().strip() and id(t) not in inactive_ticks]
        bounds=[t.get_window_extent(renderer) for t in text]
        conflicts=[]
        for i in range(len(text)):
            for j in range(i+1,len(text)):
                a,b=bounds[i],bounds[j]
                width=min(a.x1,b.x1)-max(a.x0,b.x0);height=min(a.y1,b.y1)-max(a.y0,b.y0)
                if width>.5 and height>.5:
                    conflicts.append(dict(text1=text[i].get_text(),text2=text[j].get_text(),overlap_px2=round(width*height,2)))
        records.append(dict(asset=item.name,minimum_visible_font_pt=min(t.get_fontsize() for t in text),text_intersections=conflicts))
        plt.close(item.fig)
    out=base.PROJECT/"qa/haa9_revision_figure_text_audit.json";out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(records,indent=2,ensure_ascii=False)+"\n",encoding="utf-8",newline="\n")
    base.require(all(not row["text_intersections"] for row in records),"Figure text intersections detected; inspect qa/haa9_revision_figure_text_audit.json")
    print(json.dumps(records,indent=2,ensure_ascii=False),flush=True)


def main():
    info=base.configure_style()
    bundles=[];manifest=[]
    for expected,builder in zip(base.ASSET_ORDER,builders()):
        b=builder();base.require(b.name==expected,f"Expected {expected}, got {b.name}")
        bundles.append(b);manifest.append(base.export_bundle(b,info));print("Rendered",b.name,flush=True)
    base.validate_outputs(manifest);base.write_documentation(bundles,manifest)
    audit_main_text_layout()
    print(json.dumps({"status":"ok","assets":base.ASSET_ORDER}),flush=True)


if __name__=="__main__":main()
