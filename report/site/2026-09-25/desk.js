(function(){
  var D = JSON.parse(document.getElementById('chartdata').textContent);
  var css = getComputedStyle(document.documentElement);
  function tok(n){ return css.getPropertyValue(n).trim(); }
  var fmt = function(v,d){ if(v==null) return '—'; d=(d==null?2:d); return Number(v).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d}); };
  var NS='http://www.w3.org/2000/svg';
  function el(n,a,t){ var e=document.createElementNS(NS,n); for(var k in a) e.setAttribute(k,a[k]); if(t!=null) e.textContent=t; return e; }

  var band=document.getElementById('band'); if(band){
  var tiles=[['^GSPC','标普500',0],['^RUT','罗素2000',0],['^TNX','10年期美债 %',2],['^VIX','VIX',1],['CL=F','原油 WTI',1],['GC=F','黄金',0],['DX-Y.NYB','美元指数',1]];
  tiles.forEach(function(t){
    var s=D.macro[t[0]]; if(!s||!s.close.length) return;
    var c=s.close, last=c[c.length-1], prev=c[Math.max(0,c.length-22)];
    var isY=t[0]==='^TNX'; var chg=isY?(last-prev):(last/prev-1);
    var div=document.createElement('div'); div.className='tile';
    var dtxt=isY?((chg>=0?'+':'')+chg.toFixed(2)+' pt / 月'):((chg>=0?'+':'')+(chg*100).toFixed(1)+'% / 月');
    div.innerHTML='<div class="k">'+t[1]+'</div><div class="v">'+fmt(last,t[2])+'</div><div class="d '+(chg>=0?'up':'dn')+'">'+dtxt+'</div>';
    var w=200,h=34,mn=Math.min.apply(null,c),mx=Math.max.apply(null,c),rg=(mx-mn)||1;
    var pts=c.map(function(v,i){ return [ (i/(c.length-1))*w, h-2-((v-mn)/rg)*(h-4) ]; });
    var svg=el('svg',{viewBox:'0 0 '+w+' '+h,'aria-hidden':'true',preserveAspectRatio:'none'});
    var d='M'+pts.map(function(p){return p[0].toFixed(1)+','+p[1].toFixed(1)}).join('L');
    svg.appendChild(el('path',{d:d+'L'+w+','+h+'L0,'+h+'Z',fill:tok('--accent'),'fill-opacity':'0.12'}));
    svg.appendChild(el('path',{d:d,fill:'none',stroke:tok('--accent'),'stroke-width':'1.5','vector-effect':'non-scaling-stroke'}));
    div.appendChild(svg); band.appendChild(div);
  }); }

  document.querySelectorAll('figure.chart').forEach(function(fig){
    var sym=fig.dataset.sym, s=D.stocks[sym]; if(!s) return;
    var W=760,H=300,L=8,R=64,T=14,B=28, iw=W-L-R, ih=H-T-B;
    var series=[s.close,s.sma20,s.sma50];
    var all=[].concat.apply([],series).filter(function(v){return v!=null;}).concat([s.stop,s.target,s.entry].filter(function(v){return v!=null;}));
    var mn=Math.min.apply(null,all),mx=Math.max.apply(null,all),pad=(mx-mn)*0.06; mn-=pad; mx+=pad;
    var n=s.close.length;
    var x=function(i){ return L+(i/(n-1))*iw; }, y=function(v){ return T+ih-((v-mn)/(mx-mn))*ih; };
    var legend=document.createElement('div'); legend.className='legend';
    legend.innerHTML='<span><i style="border-color:'+tok('--s1')+'"></i>收盘 '+fmt(s.close[n-1])+'</span><span><i style="border-color:'+tok('--s2')+'"></i>MA20 '+fmt(s.sma20[n-1])+'</span><span><i style="border-color:'+tok('--s3')+'"></i>MA50 '+fmt(s.sma50[n-1])+'</span><span style="color:'+tok('--ink-3')+'">'+sym+' · 近 '+n+' 个交易日 · 虚线：入场 / 止损 / 目标</span>';
    fig.appendChild(legend);
    var box=document.createElement('div'); box.className='box'; fig.appendChild(box);
    var svg=el('svg',{viewBox:'0 0 '+W+' '+H,role:'img','aria-label':sym+' 收盘价与均线'}); box.appendChild(svg);
    var ticks=4; for(var k=0;k<=ticks;k++){ var v=mn+(mx-mn)*k/ticks; svg.appendChild(el('line',{x1:L,x2:L+iw,y1:y(v),y2:y(v),stroke:tok('--rule-2'),'stroke-width':'1'})); svg.appendChild(el('text',{x:L+iw+6,y:y(v)+4,fill:tok('--ink-3')},fmt(v))); }
    [0,Math.floor(n/2),n-1].forEach(function(i,j){ svg.appendChild(el('text',{x:x(i),y:H-8,fill:tok('--ink-3'),'text-anchor':j===0?'start':(j===2?'end':'middle')},s.dates[i])); });
    [['入场',s.entry,tok('--ink-2')],['止损',s.stop,tok('--up')],['目标',s.target,tok('--dn')]].forEach(function(lv){
      if(lv[1]==null) return; var yy=y(lv[1]);
      svg.appendChild(el('line',{x1:L,x2:L+iw,y1:yy,y2:yy,stroke:lv[2],'stroke-width':'1','stroke-dasharray':'4 4','stroke-opacity':'0.8'}));
      svg.appendChild(el('text',{x:L+4,y:(lv[0]==='止损'?yy+12:yy-4),fill:lv[2],'font-size':'11'},lv[0]+' '+fmt(lv[1])));
    });
    var cols=[tok('--s1'),tok('--s2'),tok('--s3')], widths=['2','1.5','1.5'];
    [2,1,0].forEach(function(si){
      var d='',started=false;
      series[si].forEach(function(v,i){ if(v==null){started=false;return;} d+=(started?'L':'M')+x(i).toFixed(1)+','+y(v).toFixed(1); started=true; });
      if(si===0){ var area=d+'L'+x(n-1).toFixed(1)+','+(T+ih)+'L'+x(0).toFixed(1)+','+(T+ih)+'Z'; svg.appendChild(el('path',{d:area,fill:cols[0],'fill-opacity':'0.06'})); }
      svg.appendChild(el('path',{d:d,fill:'none',stroke:cols[si],'stroke-width':widths[si],'stroke-linejoin':'round'}));
    });
    svg.appendChild(el('circle',{cx:x(n-1),cy:y(s.close[n-1]),r:'3.5',fill:cols[0],stroke:tok('--surface-1'),'stroke-width':'2'}));
    var cross=el('line',{x1:0,x2:0,y1:T,y2:T+ih,stroke:tok('--ink-3'),'stroke-width':'1','stroke-dasharray':'2 3',visibility:'hidden'}); svg.appendChild(cross);
    var dot=el('circle',{r:'4',fill:cols[0],stroke:tok('--surface-1'),'stroke-width':'2',visibility:'hidden'}); svg.appendChild(dot);
    var tip=document.createElement('div'); tip.className='tip'; tip.hidden=true; box.appendChild(tip);
    function move(ev){
      var r=svg.getBoundingClientRect(); var px=(ev.clientX-r.left)*(W/r.width); var i=Math.round(((px-L)/iw)*(n-1)); i=Math.max(0,Math.min(n-1,i));
      cross.setAttribute('x1',x(i)); cross.setAttribute('x2',x(i)); cross.setAttribute('visibility','visible');
      dot.setAttribute('cx',x(i)); dot.setAttribute('cy',y(s.close[i])); dot.setAttribute('visibility','visible');
      tip.hidden=false; tip.innerHTML=s.dates[i]+'<br>收盘 '+fmt(s.close[i])+'<br>MA20 '+fmt(s.sma20[i])+' · MA50 '+fmt(s.sma50[i]);
      var left=(x(i)/W)*r.width; tip.style.left=(left+(left>r.width*0.6?-tip.offsetWidth-12:12))+'px'; tip.style.top=Math.max(0,(y(s.close[i])/H)*r.height-40)+'px';
    }
    function leave(){ cross.setAttribute('visibility','hidden'); dot.setAttribute('visibility','hidden'); tip.hidden=true; }
    svg.addEventListener('mousemove',move); svg.addEventListener('touchmove',function(e){ if(e.touches[0]) move(e.touches[0]); },{passive:true}); svg.addEventListener('mouseleave',leave);
  });
})();
