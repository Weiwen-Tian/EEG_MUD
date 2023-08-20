function plottrainingResult(x,y,model)
% DESCRIPTION
% Plot the training results
%
%    plottrainingResult(x,y,model)
%
% INPUT
%   x         training data
%   y         training data
%   model     training model
%
% Created on 5th July 2019, by Kepeng Qiu.
%-------------------------------------------------------------%


rv_index = model.rv_index;

% Delete the index of bias
if model.bias
    rv_index(model.rv_index == model.bias_index) = [];
end

figure
hold on
grid on

plot(x,y,'b:o','LineWidth',1,'MarkerSize',3, ...
    'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b')

plot(x(rv_index),y(rv_index),'ro','MarkerSize',8, ...
    'MarkerEdgeColor', 'r')

% axis settings
tgca = 12;  % font size
tfont = 'Helvetica'; % font type
% tfont = 'Arial'; % font type
% set(gca,'yscale','log')
set(gca,'FontSize',tgca,'FontName',tfont)

% legend settings
tlegend = tgca*0.9;
legend({'training samples','relevance vectors'},'FontSize',tlegend , ...
    'FontWeight','normal','FontName',tfont)

% label settings
tlabel = tgca*1.1;
xlabel('Samples','FontSize',tlabel,'FontWeight','normal', ...
    'FontName',tfont,'Color','k')
ylabel('Value','FontSize',tlabel,'FontWeight','normal', ...
    'FontName',tfont,'Color','k')

end
