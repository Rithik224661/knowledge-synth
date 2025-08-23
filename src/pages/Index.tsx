import { useState, useRef } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { QueryInput } from "@/components/research/QueryInput";
import { ResearchProcess } from "@/components/research/ResearchProcess";
import { ResultsDisplay } from "@/components/research/ResultsDisplay";
import { 
  Brain, 
  Search, 
  FileText, 
  BarChart3,
  Sparkles,
  Users,
  Target,
  Zap,
  ArrowDown
} from "lucide-react";
import heroImage from "@/assets/research-hero.jpg";

const Index = () => {
  const [isResearching, setIsResearching] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const querySectionRef = useRef<HTMLElement>(null);

  const scrollToQuery = () => {
    querySectionRef.current?.scrollIntoView({ 
      behavior: 'smooth',
      block: 'start'
    });
  };

  // Mock research process steps
  const mockSteps = [
    {
      id: "1",
      title: "Query Decomposition",
      description: "Breaking down complex query into actionable sub-tasks",
      status: "completed" as const,
      tool: "llm_analysis"
    },
    {
      id: "2", 
      title: "ArXiv Paper Search",
      description: "Searching for latest LLM safety research papers",
      status: "completed" as const,
      tool: "arxiv_search"
    },
    {
      id: "3",
      title: "Document Analysis", 
      description: "Extracting and analyzing content from research papers",
      status: "running" as const,
      tool: "document_loader"
    },
    {
      id: "4",
      title: "Dataset Identification",
      description: "Finding and evaluating mentioned datasets",
      status: "pending" as const,
      tool: "web_search"
    },
    {
      id: "5",
      title: "Trend Analysis",
      description: "Analyzing publication trends over the past year",
      status: "pending" as const,
      tool: "data_analysis"
    },
    {
      id: "6",
      title: "Final Synthesis",
      description: "Generating comprehensive research summary",
      status: "pending" as const,
      tool: "llm_synthesis"
    }
  ];

  // Mock research results
  const mockResults = {
    summary: "Based on analysis of recent LLM safety research, three key papers have emerged: 'Constitutional AI: Harmlessness from AI Feedback' (Anthropic, 2023), 'Red Teaming Language Models to Reduce Harms' (DeepMind, 2023), and 'Measuring and Narrowing the Compositionality Gap in Language Models' (OpenAI, 2023). These papers focus on alignment techniques, adversarial testing, and compositional reasoning safety. The HarmBench dataset from the first paper is recommended for its comprehensive coverage of potential harm categories.",
    sources: [
      {
        title: "Constitutional AI: Harmlessness from AI Feedback",
        url: "https://arxiv.org/abs/2212.08073",
        type: "paper" as const,
        relevanceScore: 0.95
      },
      {
        title: "Red Teaming Language Models to Reduce Harms",
        url: "https://arxiv.org/abs/2209.07858", 
        type: "paper" as const,
        relevanceScore: 0.88
      },
      {
        title: "HarmBench: A Standardized Evaluation Framework",
        url: "https://github.com/center-for-ai-safety/HarmBench",
        type: "dataset" as const,
        relevanceScore: 0.92
      }
    ],
    recommendations: [
      "Use HarmBench dataset for comprehensive safety evaluation - contains 510 adversarial prompts across 18 harm categories",
      "Implement Constitutional AI techniques for self-supervision in safety alignment",
      "Consider red-teaming approaches early in model development lifecycle"
    ],
    evaluation: [
      {
        metric: "Answer Faithfulness",
        score: 0.89,
        description: "Generated answer is well-grounded in retrieved sources with minimal hallucination"
      },
      {
        metric: "Answer Relevance", 
        score: 0.94,
        description: "Response directly addresses all parts of the original query comprehensively"
      },
      {
        metric: "Tool Use Accuracy",
        score: 0.87,
        description: "Agent selected appropriate tools and parameters for each sub-task"
      }
    ]
  };

  const handleQuerySubmit = (query: string) => {
    setIsResearching(true);
    // Simulate research process
    setTimeout(() => {
      setShowResults(true);
      setIsResearching(false);
    }, 3000);
  };

  const features = [
    {
      icon: <Brain className="h-6 w-6" />,
      title: "Multi-Tool Orchestration",
      description: "Intelligently coordinates web search, document analysis, and LLM synthesis"
    },
    {
      icon: <Target className="h-6 w-6" />,
      title: "Query Decomposition", 
      description: "Breaks complex requests into actionable sub-tasks using advanced reasoning"
    },
    {
      icon: <BarChart3 className="h-6 w-6" />,
      title: "Performance Evaluation",
      description: "Built-in faithfulness, relevance, and tool accuracy assessment"
    },
    {
      icon: <Zap className="h-6 w-6" />,
      title: "Real-time Processing",
      description: "Live updates on research progress with transparent tool execution"
    }
  ];

  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-hero min-h-screen flex items-center">
        <div className="absolute inset-0">
          <img 
            src={heroImage} 
            alt="AI Research Assistant Platform" 
            className="w-full h-full object-cover opacity-15"
          />
          <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-background/20 to-background/40"></div>
        </div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
          <div className="text-center space-y-8 animate-fade-up">
            <div className="space-y-6">
              <Badge variant="outline" className="mb-6 px-4 py-2 text-sm font-medium animate-scale-in" style={{animationDelay: '0.2s'}}>
                <Sparkles className="h-4 w-4 mr-2" />
                AI-Powered Research Assistant
              </Badge>
              <h1 className="text-hero leading-tight" style={{animationDelay: '0.4s'}}>
                Multi-Tool AI Research
                <span className="block bg-gradient-research bg-clip-text text-transparent">Assistant</span>
              </h1>
              <p className="text-subtitle max-w-4xl mx-auto leading-relaxed" style={{animationDelay: '0.6s'}}>
                Advanced agent framework that orchestrates web search, document analysis, and LLM synthesis 
                to deliver comprehensive, evaluated research insights for founders and scientists.
              </p>
            </div>
            
            <div className="flex flex-wrap justify-center gap-6 animate-slide-up" style={{animationDelay: '0.8s'}}>
              <Button 
                variant="research" 
                size="lg" 
                onClick={scrollToQuery}
                className="btn-research px-8 py-4 text-lg font-semibold"
              >
                <Brain className="h-6 w-6 mr-3" />
                Start Research
              </Button>
              <Button variant="outline" size="lg" className="px-8 py-4 text-lg font-semibold">
                <FileText className="h-6 w-6 mr-3" />
                View Documentation
              </Button>
            </div>
            
            {/* Scroll Indicator */}
            <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce">
              <button 
                onClick={scrollToQuery}
                className="p-2 rounded-full bg-primary/10 border border-primary/20 hover:bg-primary/20 transition-all duration-300"
              >
                <ArrowDown className="h-6 w-6 text-primary" />
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-muted/30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16 animate-fade-up">
            <h2 className="text-4xl font-bold text-foreground mb-6">Core Capabilities</h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
              Built on LangChain with sophisticated orchestration, evaluation, and multi-tool integration
            </p>
          </div>
          
          <div className="feature-grid">
            {features.map((feature, index) => (
              <Card key={index} className="p-8 text-center card-elevated animate-scale-in" style={{animationDelay: `${index * 0.1}s`}}>
                <div className="w-16 h-16 bg-gradient-research rounded-xl flex items-center justify-center mx-auto mb-6 text-primary-foreground">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-semibold text-foreground mb-4">{feature.title}</h3>
                <p className="text-muted-foreground leading-relaxed">{feature.description}</p>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Main Interface */}
      <section ref={querySectionRef} className="py-20 scroll-target">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12 animate-fade-up">
            <h2 className="text-4xl font-bold text-foreground mb-4">Research Interface</h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Enter your complex research query and watch our multi-tool AI agent work
            </p>
          </div>
          
          <div className="grid lg:grid-cols-2 gap-12">
            <div className="space-y-8 animate-slide-up">
              <QueryInput onSubmit={handleQuerySubmit} isLoading={isResearching} />
              
              {(isResearching || showResults) && (
                <div className="animate-fade-up">
                  <ResearchProcess 
                    steps={mockSteps} 
                    currentStep={isResearching ? "3" : undefined}
                  />
                </div>
              )}
            </div>

            <div className="space-y-8 animate-slide-up" style={{animationDelay: '0.2s'}}>
              {showResults ? (
                <div className="animate-scale-in">
                  <ResultsDisplay result={mockResults} />
                </div>
              ) : (
                <Card className="p-12 card-elevated text-center">
                  <div className="max-w-md mx-auto">
                    <Search className="h-16 w-16 text-primary mx-auto mb-6 opacity-60" />
                    <h3 className="text-2xl font-semibold text-foreground mb-4">Ready for Research</h3>
                    <p className="text-muted-foreground leading-relaxed">
                      Submit a complex research query to see the multi-tool agent in action.
                      Results will appear here with full evaluation metrics.
                    </p>
                  </div>
                </Card>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* Backend Notice */}
      <section className="py-16 bg-accent/20">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="p-8 card-elevated animate-fade-up">
            <div className="w-16 h-16 bg-gradient-research rounded-xl flex items-center justify-center mx-auto mb-6">
              <Users className="h-8 w-8 text-primary-foreground" />
            </div>
            <h3 className="text-2xl font-semibold text-foreground mb-4">Backend Integration Required</h3>
            <p className="text-lg text-muted-foreground mb-6 leading-relaxed">
              To enable full functionality including LangChain orchestration, LLM calls, web search, 
              and document processing, connect this project to Supabase using our native integration.
            </p>
            <p className="text-muted-foreground">
              Click the green Supabase button in the top right to get started with backend functionality.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Index;