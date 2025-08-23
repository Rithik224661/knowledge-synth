import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  FileText, 
  BarChart3, 
  CheckCircle, 
  Download,
  ExternalLink,
  Star
} from "lucide-react";

interface Source {
  title: string;
  url: string;
  type: "paper" | "article" | "dataset";
  relevanceScore: number;
}

interface Evaluation {
  metric: string;
  score: number;
  description: string;
}

interface ResearchResult {
  summary: string;
  sources: Source[];
  recommendations: string[];
  evaluation: Evaluation[];
  visualizationData?: any;
}

interface ResultsDisplayProps {
  result: ResearchResult;
}

export const ResultsDisplay = ({ result }: ResultsDisplayProps) => {
  const getSourceIcon = (type: string) => {
    switch (type) {
      case "paper":
        return <FileText className="h-4 w-4" />;
      case "dataset":
        return <BarChart3 className="h-4 w-4" />;
      default:
        return <ExternalLink className="h-4 w-4" />;
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return "text-success";
    if (score >= 0.6) return "text-warning";
    return "text-destructive";
  };

  return (
    <Card className="p-6 bg-gradient-card shadow-elevated">
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-foreground">Research Results</h2>
          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Export Report
          </Button>
        </div>

        <Tabs defaultValue="summary" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="summary">Summary</TabsTrigger>
            <TabsTrigger value="sources">Sources</TabsTrigger>
            <TabsTrigger value="evaluation">Evaluation</TabsTrigger>
            <TabsTrigger value="visualization">Charts</TabsTrigger>
          </TabsList>

          <TabsContent value="summary" className="space-y-4">
            <div className="prose prose-sm max-w-none">
              <div className="p-4 bg-background rounded-lg border">
                <p className="text-foreground leading-relaxed">{result.summary}</p>
              </div>
            </div>

            {result.recommendations.length > 0 && (
              <div className="space-y-3">
                <h3 className="font-medium text-foreground flex items-center gap-2">
                  <Star className="h-4 w-4 text-primary" />
                  Key Recommendations
                </h3>
                <div className="space-y-2">
                  {result.recommendations.map((rec, index) => (
                    <div key={index} className="flex items-start gap-3 p-3 bg-accent/30 rounded-lg">
                      <CheckCircle className="h-4 w-4 text-success mt-0.5 flex-shrink-0" />
                      <p className="text-sm text-foreground">{rec}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </TabsContent>

          <TabsContent value="sources" className="space-y-4">
            <div className="grid gap-4">
              {result.sources.map((source, index) => (
                <div key={index} className="p-4 bg-background rounded-lg border hover:shadow-card transition-shadow">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        {getSourceIcon(source.type)}
                        <h4 className="font-medium text-foreground text-sm">{source.title}</h4>
                        <Badge variant="outline" className="text-xs">
                          {source.type}
                        </Badge>
                      </div>
                      <a 
                        href={source.url} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="text-xs text-primary hover:underline"
                      >
                        {source.url}
                      </a>
                    </div>
                    <div className="text-right">
                      <div className="text-xs text-muted-foreground">Relevance</div>
                      <div className={`text-sm font-medium ${getScoreColor(source.relevanceScore)}`}>
                        {(source.relevanceScore * 100).toFixed(0)}%
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="evaluation" className="space-y-4">
          <div className="grid gap-4">
              {result.evaluation.map((evaluation, index) => (
                <Card key={index} className="p-4">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium text-foreground">{evaluation.metric}</h4>
                    <div className={`text-lg font-bold ${getScoreColor(evaluation.score)}`}>
                      {(evaluation.score * 100).toFixed(0)}%
                    </div>
                  </div>
                  <p className="text-sm text-muted-foreground">{evaluation.description}</p>
                  <div className="mt-2 w-full bg-muted rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full transition-all ${
                        evaluation.score >= 0.8 ? "bg-success" : 
                        evaluation.score >= 0.6 ? "bg-warning" : "bg-destructive"
                      }`}
                      style={{ width: `${evaluation.score * 100}%` }}
                    />
                  </div>
                </Card>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="visualization" className="space-y-4">
            <div className="p-8 bg-background rounded-lg border text-center">
              <BarChart3 className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="font-medium text-foreground mb-2">Visualization Coming Soon</h3>
              <p className="text-sm text-muted-foreground">
                Interactive charts and graphs will be displayed here once the backend is connected.
              </p>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </Card>
  );
};