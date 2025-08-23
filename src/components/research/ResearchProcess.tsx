import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { 
  Search, 
  FileText, 
  Brain, 
  BarChart3, 
  CheckCircle, 
  Clock,
  AlertCircle
} from "lucide-react";

interface ProcessStep {
  id: string;
  title: string;
  description: string;
  status: "pending" | "running" | "completed" | "error";
  tool: string;
}

interface ResearchProcessProps {
  steps: ProcessStep[];
  currentStep?: string;
}

export const ResearchProcess = ({ steps, currentStep }: ResearchProcessProps) => {
  const getStepIcon = (tool: string) => {
    switch (tool) {
      case "web_search":
      case "arxiv_search":
        return <Search className="h-4 w-4" />;
      case "document_loader":
      case "pdf_parser":
        return <FileText className="h-4 w-4" />;
      case "llm_synthesis":
      case "analysis":
        return <Brain className="h-4 w-4" />;
      case "visualization":
      case "data_analysis":
        return <BarChart3 className="h-4 w-4" />;
      default:
        return <Search className="h-4 w-4" />;
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="h-4 w-4 text-success" />;
      case "running":
        return <Clock className="h-4 w-4 text-warning animate-spin" />;
      case "error":
        return <AlertCircle className="h-4 w-4 text-destructive" />;
      default:
        return <Clock className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "completed":
        return <Badge variant="success">Completed</Badge>;
      case "running":
        return <Badge variant="warning">Running</Badge>;
      case "error":
        return <Badge variant="destructive">Error</Badge>;
      default:
        return <Badge variant="secondary">Pending</Badge>;
    }
  };

  const completedSteps = steps.filter(step => step.status === "completed").length;
  const progress = (completedSteps / steps.length) * 100;

  return (
    <Card className="p-6 bg-gradient-card shadow-card">
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-foreground">Research Process</h2>
          <div className="text-sm text-muted-foreground">
            {completedSteps}/{steps.length} steps completed
          </div>
        </div>

        <Progress value={progress} className="w-full" />

        <div className="space-y-4">
          {steps.map((step, index) => (
            <div
              key={step.id}
              className={`flex items-start gap-4 p-4 rounded-lg border transition-all ${
                step.id === currentStep
                  ? "border-primary bg-accent/50"
                  : "border-border bg-background/50"
              }`}
            >
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-muted flex items-center justify-center">
                {getStepIcon(step.tool)}
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <h3 className="font-medium text-foreground">{step.title}</h3>
                  {getStatusBadge(step.status)}
                </div>
                <p className="text-sm text-muted-foreground mb-2">{step.description}</p>
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className="text-xs">
                    {step.tool}
                  </Badge>
                </div>
              </div>

              <div className="flex-shrink-0">
                {getStatusIcon(step.status)}
              </div>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
};