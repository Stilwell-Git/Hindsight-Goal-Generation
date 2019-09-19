#include <stdio.h>
#include <stdlib.h>

#define inf 1000000000
#define bool int
#define true 1
#define false 0

#define Graph_vertex 100000
#define Graph_edge 20000000
#define flow_value int
#define cost_value long long
#define dist_inf 1000000000000000ll

int n,i,tot,u,v,ll,rr;
int head[Graph_vertex],next[Graph_edge],ed[Graph_edge],from[Graph_vertex],q[Graph_vertex];
flow_value flow[Graph_edge],max_flow,sum_flow;
cost_value dist[Graph_vertex],cost[Graph_edge],ans;
bool inq[Graph_vertex];

void clear(int new_n)
{
	n=new_n;tot=1;
	for(i=0;i<=n;++i)head[i]=0;
}
int add(int u,int v,flow_value f,double c_float)
{
	long long c=(long long)(c_float*1000000);
	next[++tot]=head[u];head[u]=tot;ed[tot]=v;flow[tot]=f;cost[tot]=c;
	next[++tot]=head[v];head[v]=tot;ed[tot]=u;flow[tot]=0;cost[tot]=-c;
	return tot;
}

// https://en.wikipedia.org/wiki/Shortest_Path_Faster_Algorithm
void SPFA(int s)
{
	for(i=0;i<=n;++i)dist[i]=dist_inf;
	ll=0;q[rr=1]=s;dist[s]=0;
	while(ll!=rr)
	{
		++ll;if(ll==Graph_vertex)ll=1;
		u=q[ll];inq[u]=false;
		for(i=head[u];i;i=next[i])
		if(flow[i]&&dist[u]+cost[i]<dist[v=ed[i]])
		{
			dist[v]=dist[u]+cost[from[v]=i];
			if(!inq[v])
			{
				++rr;if(rr==Graph_vertex)rr=1;
				q[rr]=v;inq[v]=true;
			}
		}
	}
}

// https://en.wikipedia.org/wiki/Minimum-cost_flow_problem
flow_value cost_flow(int s,int t)
{
	ans=0;sum_flow=0;
	for(;;)
	{
		SPFA(s);
		if(dist[t]>=dist_inf)break;
		max_flow=inf;
		for(u=t;u!=s;u=ed[i^1])
		{
			i=from[u];
			if(flow[i]<max_flow)max_flow=flow[i];
		}
		sum_flow+=max_flow;
		ans+=dist[t]*max_flow;
		for(u=t;u!=s;u=ed[i^1])
		{
			i=from[u];
			flow[i]-=max_flow;
			flow[i^1]+=max_flow;
		}
	}
	return sum_flow;
}

bool check_match(int edge_id)
{
	if(flow[edge_id]==0)return false;
	return true;
}

#undef Graph_vertex
#undef Graph_edge
#undef flow_value
#undef cost_value
#undef dist_inf
