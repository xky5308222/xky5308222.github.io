##                                                                 NOIP PJ Template

### Fast Read&Print

```c++
inline char nc(){
  static char buf[100000],*p1=buf,*p2=buf;
  if (p1==p2){ 
      p2=(p1=buf)+fread(buf,1,100000,stdin); 
      if (p1==p2) return EOF; 
  }
  return *p1++;
}

inline int read(){
  char c=nc();int b=1,x=0;
  for(;!(c>='0' && c<='9');c=nc()) if (c=='-') b=-1;
  for(x=0;c>='0' && c<='9';x=x*10+c-'0',c=nc()); x*=b;
  return x;
}

int wt,ss[19];
inline void print(int x){
	if (x<0) x=-x,putchar('-'); 
	if (!x) putchar(48); 
    else{
		for(wt=0;x;ss[++wt]=x%10,x/=10);
		for(;wt;putchar(ss[wt]+48),wt--);
    }
}
```

### Heap

```c++
void put(int x){
    h[++l]=x;
    int s=l,t;
    while(s!=1 && h[s>>1]>h[s]){
        t=h[s>>1],h[s>>1]=h[s],h[s]=t;
        s/=2;
    }
}

int get(){
    int res=h[1];
    h[1]=h[l],l--;
    int f=1,t;
    while(f*2<=l){
        if (f*2+1>len || h[f*2]<h[f*2+1]) s=f*2;
        else s=f*2+1;
        if (h[f]>h[s]) t=h[f],h[f]=h[s],h[s]=t,f=s;
        else break;
    }
    return res;
}
```

### Union Find

```c++
int gf(int k){
    if (f[k]==k) return k;
    f[k]=gf(f[k]);
    return f[k];
}

void uni(int x,int y){
    int f1=gf(x),f2=gf(y);
    f[f1]=f2;
}

bool judge(int x,int y){
    int f1=gf(x),f2=gf(y);
    return f1==f2;
}
```

### Quicksort

```c++
void sort(int l,int r){
    int i=l,j=r,x=a[(l+r)>>1],t;
    do{
        while(a[i]<x) i++;
        while(x<a[j]) j++;
        if (i<=j) t=a[i],a[i]=a[j],a[j]=t,i++,j--;
    }while(i<=j);
    if (l<j) sort(l,j);
    if (i<r) sort(i,r);
}
```

### 埃氏筛

```c++
f[0]=f[1]=true;
for(int i=2;i<=n;i++)
    if (!f[i]){
        prime[++cnt]=i;
        int j=2;
        while(i*j<=n) f[i*j]=true,j++;
    }
```

### 欧拉筛

```c++
f[0]=f[1]=true;
for(int i=2;i<=n;i++){
    if (!f[i]) prime[++cnt]=i;
    int j=1;
    while(i*prime[j]<=n && j<=cnt){
        f[i*prime[j]]=true;
        if (i%prime[j]==0) break;
        j++;
    }
}
```

### ST Form

```c++
void llog(){
    for(int i=1;i<=n;i++) lg[i]=lg[i/2]+1;
    for(int i=1;i<=n;i++) --lg[i];
}

void work(){
    for(int i=1;i<=n;i++) f[i][0]=a[i];
    int i=1;
    while((1<<i)<=n){
        int j=1;
        while(j+(1<<i)-1<=n) f[j][i]=max(f[j][i-1],f[j+1 shl (i-1)][i-1]),++j;
        ++i;
    }
}

int query(int l,int r){
    int k=lg[r-l+1];
    return max(f[l][k],f[r-(1<<k)+1][k]);
}
```

### BIT

```c++
int lowbit(int x){
    return x&-x;
}

void change(int x,int y){
    int t=x;
    while(t<=n) cnt+=y,t+=lowbit(t);
}

int query(int x){
    int t=x,ans=0;
    while(t>0) ans+=c[t],t-=lowbit(t);
    return ans;
}
```

#### 单点修改&区间查询

```c++
int n=read(),m=read();
for(int i=1;i<=n;i++)
    a[i]=read(),change(i,a[i]);
for(int i=1;i<=m;i++){
    int num=read(),x=read(),y=read();
    if (num==1) change(x,y);
    else printf("%d\n",query(y)-query(x-1));
}
```

#### 区间修改+单点查询

```c++
int n=read(),m=read();
for(int i=1;i<=n;i++) a[i]=read();
for(int i=1;i<=m;i++){
    int num=read();
    if (num==1) {
        int x=read(),y=read(),s=read();
        change(x,s);
        change(y+1,-s);//差分
    }
    else{
        int x=read();
        printf("%d\n",a[x]+query(x));//差分
    }
}
```

### Segment Tree

```c++
struct node{
    int sum,m,tag;
}tree[4*maxn];

void lc(int x){ return x<<1; }
void rc(int x){ return (x<<1)+1; }

void pushup(int x){
    tree[x].sum=tree[l(x)].sum+tree[r(x)].sum;
    tree[x].m=max(tree[l(x)].m,tree[r(x)].m);
}

void build(int l,int r,int x){
    if (l==r){
        tree[x].sum=a[l],tree[x].m=a[l];
        return;
    }
    int mid=(l+r)>>1;
    build(l,mid,lc(x));
    build(mid+1,r,rc(x));
    pushup(x);
}

void pushdown(int x,int l,int r){
    int mid=(l+r)>>1;
    if (tree[x].tag>0){
        tree[lc(x)].tag+=tree[x].tag;
        tree[rc(x)].tag+=tree[x].tag;
        tree[lc(x)].sum+=tree[x].tag*(mid-l+1);
        tree[rc(x)].sum+=tree[x].tag*(r-mid);
        tree[lc(x)].m+=tree[x].tag;
        tree[rc(x)].m+=tree[x].tag;
    }
}

void change(int lq,int rq,int l,int r,int x,int y){
    if (lq<=l && rq>=r){
        tree[x].tag+=y;
        tree[x].sum+=y*(r-l+1);
        tree[x].m+=y;
        return;
    }
    if (lq>r || rq<l) return;
    int mid=(l+r)>>1;
    if (lq<=mid) change(lq,rq,l,mid,lc(x),y);
    if (rq>mid) change(lq,rq,mid+1,r,rc(x),y);
}

int querysum(int lq,int rq,int l,int r,int x){
    if (lq<=l && rq>=r) return tree[x].sum;
    if (lq>r || rq<l) return 0;
    int mid=(l+r)>>1,ans=0;
    pushdown(x,l,r);
    if (lq<=mid) ans+=querysum(lq,rq,l,mid,lc(x));
    if (rq>mid) ans+=querysum(lq,rq,mid+1,r,rc(x));
    pushup(x);
    return ans;
}

int querymax(int lq,int rq,int l,int r,int x){
    if (lq<=l && rq>=r) return tree[x].m;
    if (lq>r || rq<l) return 0;
    int mid=(l+r)>>1,ans=0;
    pushdown(x,l,r);
    if (lq<=mid) ans=max(ans,querymax(lq,rq,l,mid,lc(x)));
    if (rq>mid) ans=max(ans,querymax(lq,rq,mid+1,r,rc(x)));
    pushup(x);
    return ans;
}
//build(1,n,1);
//change(l,r,1,n,1,x);
//querysum(l,r,1,n,1);
//querymax(l,r,1,n,1);
```

### MST(Kruscal)

```c++
#include <bits/stdc++.h>
using namespace std;

struct node{
    int x,y,l;
}a[2000010];
int f[2000010];

void cmp(node x,node y){
    return x.l<y.l;
}

int gf(int k){
    if (f[k]==k) return k;
    f[k]=gf(f[k]);
    return f[k];
}

void uni(int x,int y){
    int f1=gf(x),f2=gf(y);
    f[f1]=f2;
}

int main()
{
    int n=read(),m=read();
    for(int i=1;i<=n;i++)
        a[i].u=read(),a[i].v=read(),a[i].l=read();
    sort(a+1,a+n+1,cmp);
    for(int i=1;i<=n;i++)
        f[i]=i;
    int ans=0;
    for(int i=1;i<=m;i++)
        if (gf(a[i].u)!=gf(a[i].v)){
            uni(a[i].u,a[i].v),cnt++,ans+=a[i].l;
            if (cnt==n-1) break;
        }
    print(ans);
}
```

### Dijkstra

```c++
#include <bits/stdc++.h>
#define ll long long
using namespace std;

ll head[10010],ver[500010],edge[500010],Next[500010],d[500010];
bool v[10010];
ll n,m,tot,s;
priority_queue<pair<ll,ll> >q;

inline ll read(){
    ll tmp=1,x=0;
    char ch=getchar();
    while(!isdigit(ch)){
        if(ch=='-') tmp=-1;
        ch=getchar();
    }
    while(isdigit(ch)){
        x=x*10+ch-48;
        ch=getchar();
    }
    return tmp*x;
}

inline void addEdge(ll x,ll y,ll z){
    ver[++tot]=y;
    edge[tot]=z;
    Next[tot]=head[x];
    head[x]=tot;
}

void dijkstra(){
    for(ll i=1; i<=500005; i++)d[i]=2147483647;
    memset(v,0,sizeof(v));
    d[s]=0;
    q.push(make_pair(0,s));
    while(q.size()){
        ll x=q.top().second; q.pop();
        if(v[x]) continue;
        v[x]=1;
        for(ll i=head[x]; i; i=Next[i]){
            ll y=ver[i],z=edge[i];
            if(d[y]>d[x]+z){
                d[y]=d[x]+z;
                q.push(make_pair(-d[y],y));
            }
        }
    }
}

int main(){
    n=read(); m=read(); s=read();
    for(ll i=1; i<=m; i++){
        ll x=read(),y=read(),z=read();
        addEdge(x,y,z);
    }
    dijkstra();
    for(ll i=1; i<=n; i++){
        printf("%lld ",d[i]);
    }
}
```

### SPFA

考场慎用，容易退化到$O(NM)$.~~关于$SPFA$:它死了~~

```c++
#include<bits/stdc++.h>
const long long inf=2147483647;
const int maxn=10005;
const int maxm=500005;
using namespace std;

int n,m,s,num_edge=0;
int dis[maxn],vis[maxn],head[maxm];
struct Edge
{
  int next,to,dis;
}edge[maxm];

void addedge(int from,int to,int dis)
{
  edge[++num_edge].next=head[from];
  edge[num_edge].to=to;
  edge[num_edge].dis=dis;
  head[from]=num_edge;
}

void spfa()
{
  	queue<int> q;
  	for(int i=1; i<=n; i++) 
		dis[i]=inf,vis[i]=0;
  	q.push(s),dis[s]=0,vis[s]=1;
 	while(!q.empty())
  	{
    	int u=q.front();
    	q.pop(); vis[u]=0;
    	for(int i=head[u]; i; i=edge[i].next)
    	{
      		int v=edge[i].to; 
      		if(dis[v]>dis[u]+edge[i].dis)
      		{
        		dis[v]=dis[u]+edge[i].dis;
        		if(vis[v]==0) vis[v]=1,q.push(v);
      		}
    	}
  	}
}

int main()
{
  	cin>>n>>m>>s;
  	for(int i=1; i<=m; i++)
  	{
    	int f,g,w;
    	cin>>f>>g>>w; 
    	addedge(f,g,w);
  	}
  	spfa();
  	for(int i=1; i<=n; i++)
    	if(s==i) cout<<0<<" ";
        else cout<<dis[i]<<" ";
  	return 0;
}
```

### Quickpower

```c++
while(m>0){
    if(m%2==1) ans=ans*b%p;
    b=b*b%p;
    m=m>>1;
}
printf("%lld",ans%p);//b^m % p.
```

### 乘法逆元(exgcd)

```c++
//标准版
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
ll x,y;

void exgcd(ll a,ll b){
	if (b==0){
		x=1,y=0;
		return;
	}
	exgcd(b,a%b);
	ll tx=x;
	x=y,y=tx-(a/b)*y;
}

int main()
{
	ll a,b;
	scanf("%lld%lld",&a,&b);
	for(ll i=1;i<=a;i++){
		exgcd(i,b);
		x=(x%b+b)%b;
		printf("%lld\n",x);
	}
	return 0;
}
//线性递推
#include <bits/stdc++.h>
using namespace std;

long long in[3000010];

int main()
{
	int a,b;
	scanf("%d%d",&a,&b);
	printf("1\n");
	in[0]=0,in[1]=1;
	for(int i=2;i<=a;i++)
		in[i]=b-(b/i)*in[b%i]%b,printf("%d\n",in[i]);
	return 0;
}
```

### 单调栈&单调队列

```c++
//单调栈
#include <bits/stdc++.h>
#define mp make_pair
using namespace std;
stack<long long> S;
long long a[5000005],ans;
int main(){
    long long n;
    scanf("%lld",&n);
    for(long long i=1;i<=n;++i) scanf("%lld",&a[i]);
    a[n+1]=10086001100860011ll;//结束标识符
    for(long long i=1;i<=n+1;++i)
    {
        if(S.empty() || a[S.top()]>=a[i])   S.push(i);
        else
        {
            while(S.size() && a[S.top()]<a[i])
            {
                long long Top=S.top();
                ans+=(i-Top-1);
                S.pop();
            }
            S.push(i);
        }
    }
    printf("%lld",ans);
    return 0;
}
//单调队列
#include <bits/stdc++.h>
using namespace std;
int n,m;
int q1[1000001],q2[1000001];
int a[1000001];
int min_deque()
{
    int h=1,t=0;
    for(int i=1;i<=n;i++)
    {
        while(h<=t&&q1[h]+m<=i) h++;
        while(h<=t&&a[i]<a[q1[t]]) t--;
        q1[++t]=i;
        if(i>=m) printf("%d ",a[q1[h]]);
    }
    cout<<endl;
}
int max_deque()
{
    int h=1,t=0;
    for(int i=1;i<=n;i++)
    {
        while(h<=t&&q2[h]+m<=i) h++;
        while(h<=t&&a[i]>a[q2[t]]) t--;
        q2[++t]=i;
        if(i>=m) printf("%d ",a[q2[h]]);
    }
}
int main()
{
    cin>>n>>m;
    for(int i=1;i<=n;i++) scanf("%d",&a[i]);
    min_deque();
    max_deque();
    return 0;
}
```

